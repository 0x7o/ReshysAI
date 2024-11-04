import os
import random
import threading
import time

import pandas as pd
import numpy as np
from typing import List, Callable, Tuple
from dataclasses import dataclass
import concurrent.futures
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from openai import OpenAI

client = OpenAI(
    api_key="sk-or-...",
    base_url="https://openrouter.ai/api/v1",
)


@dataclass
class Program:
    code: str
    score: float
    generation: int
    parent_ids: List[str]
    program_id: str

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "score": self.score,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "program_id": self.program_id,
        }

    @staticmethod
    def from_dict(data: dict) -> "Program":
        return Program(
            code=data["code"],
            score=data["score"],
            generation=data["generation"],
            parent_ids=data["parent_ids"],
            program_id=data["program_id"],
        )


class Island:
    def __init__(self, island_id: int):
        self.island_id = island_id
        self.best_program = None
        self.programs = []

    def register_program(self, program: Program):
        """Registers a program on the island."""
        self.programs.append(program)
        if self.best_program is None or program.score < self.best_program.score:
            self.best_program = program
            print(f"Best program updated for island {self.island_id}: {program.score}")
            self._save_best_program()

    def _save_best_program(self):
        """Saves the best program of the island."""
        if self.best_program is not None:
            with open(f"best_program_{self.island_id}.py", "w") as f:
                f.write(str(self.best_program.code))

    def sample(self, k=2):
        """Samples k parents from the population."""
        if len(self.programs) < k:
            return self.programs

        tournament_size = min(5, len(self.programs))
        parents = []

        for _ in range(k):
            tournament = np.random.choice(self.programs, tournament_size, replace=False)
            winner = min(tournament, key=lambda x: x.score)
            parents.append(winner)

        # Sort parents by score (from highest to lowest)
        parents.sort(key=lambda x: x.score, reverse=True)
        return parents


class FunSearch:
    def __init__(
            self,
            eval_function: Callable,
            llm_pipeline,
            initial_program: str,
            price_limit: float = 1,
            model: str = "openai/gpt-4o-mini",
            islands: int = 10,
            #programs_per_island: int = 100,
            diversity_threshold: float = 0.3,
            parallel_workers: int = 1,
            pool_file: str = "program_pool.json",
    ):
        self.eval_function = eval_function
        self.llm_pipeline = llm_pipeline
        self.model = model
        self.islands = islands
        #self.programs_per_island = programs_per_island
        self.diversity_threshold = diversity_threshold
        self.parallel_workers = parallel_workers
        self.program_pool: List[Program] = []
        self.price_limit = price_limit
        self.generation = 0
        self.total_price = 0
        self.prompt_prices = {
            "openai/gpt-4o": 2.5,
            "openai/gpt-4o-mini": 0.15,
            "deepseek/deepseek-chat": 0.14,
            "anthropic/claude-3.5-sonnet": 3,
            "openai/o1-mini": 3,
        }
        self.completion_prices = {
            "openai/gpt-4o": 10,
            "openai/gpt-4o-mini": 0.6,
            "deepseek/deepseek-chat": 0.28,
            "anthropic/claude-3.5-sonnet": 15,
            "openai/o1-mini": 12,
        }
        self.prompt_price = self.prompt_prices[model]
        self.completion_price = self.completion_prices[model]
        self.pool_file = pool_file

        self.lock = threading.Lock()

        # Initialize timing variables
        self.start_time = time.time()
        self.last_reset_time = self.start_time
        self.reset_interval = 60 * 60  # 1 hour

        self.islands = [
            Island(i) for i in range(self.islands)
        ]

        # Load best programs for each island
        for idx, island in enumerate(self.islands):
            filepath = f"best_program_{idx}.py"
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    code = f.read()
                score = self._evaluate_program(code)
                program = Program(
                    code=code,
                    score=score,
                    generation=0,
                    parent_ids=[],
                    program_id=f"loaded_{idx}",
                )
                island.register_program(program)

        # Evaluate seed program
        score = self._evaluate_program(initial_program)
        if score is not None:
            initial_program = Program(
                code=initial_program,
                score=score,
                generation=0,
                parent_ids=[],
                program_id="seed",
            )
            print(f"Initial program score: {score}")
            # Register initial programs
            for island in self.islands:
                if len(island.programs) == 0:
                    island.register_program(initial_program)

    def _evaluate_program(self, program_code: str) -> float:
        """Safely evaluate a program and return its score"""
        try:
            namespace = {}
            program_code = "import numpy as np\nimport pandas as pd\n\n" + program_code
            exec(program_code, namespace)
            if "formula" not in namespace:
                return None
            return self.eval_function(namespace["formula"])
        except Exception as e:
            return None

    def _calculate_diversity(self, program: Program, island_id: int) -> float:
        """Calculate diversity score for a program compared to pool"""
        if not self.islands[island_id].programs:
            return 1.0

        # Simple diversity measure based on code similarity
        similarities = []
        for pool_program in self.islands[island_id].programs:
            similarity = len(
                set(program.code.split()) & set(pool_program.code.split())
            ) / len(set(program.code.split()) | set(pool_program.code.split()))
            similarities.append(similarity)

        return 1 - max(similarities)

    def _generate_prompt(self, parents: List[Program]) -> str:
        """Generate prompt for LLM based on parent programs"""
        programs_str = "\n\n".join(
            [
                f"Program (Score: {p.score:.3f}):\n```python\n{p.code}```"
                for p in parents
            ]
        )

        with open("prompt", "r") as file:
            prompt = file.read()

        prompt = prompt.replace("{programs_str}", programs_str)
        return prompt

    def evolve(self) -> Program:
        """Main evolution loop with parallel evaluation and periodic resets"""
        while True:
            current_time = time.time()

            if current_time - self.last_reset_time >= self.reset_interval:
                self._perform_reset()
                self.last_reset_time = current_time  # Update the last reset time

            self.generation += 1

            tasks = []
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.parallel_workers
            ) as executor:
                for idx, island in enumerate(self.islands):
                    parents = island.sample(k=2)
                    prompt = self._generate_prompt(parents)
                    task = executor.submit(
                        self._generate_and_evaluate_candidate, prompt, parents, idx
                    )
                    tasks.append(task)

                for future in concurrent.futures.as_completed(tasks):
                    result = future.result()
                    if result is not None:
                        program, island_id = result
                        if program is not None:
                            diversity = self._calculate_diversity(program, island_id)
                            with self.lock:
                                if (
                                        diversity >= self.diversity_threshold
                                        or program.score
                                        < min(p.score for p in self.islands[island_id].programs)
                                ):
                                    self.islands[island_id].register_program(program)

            if self.total_price > self.price_limit:
                print(f"Reached price limit of {self.price_limit}$. Stopped")
                break

        best_program = min(
            (
                island.best_program
                for island in self.islands
                if island.best_program is not None
            ),
            key=lambda x: x.score,
            default=None,
        )
        return best_program

    def _perform_reset(self):
        """Performs the periodic reset of half the islands"""
        with self.lock:
            m = len(self.islands)
            num_to_reset = m // 2

            # Sort islands by their best_program.score (ascending order; lower is better)
            sorted_islands = sorted(
                self.islands,
                key=lambda island: island.best_program.score
                if island.best_program
                else float("inf"),
            )

            # Identify the worst m/2 islands
            worst_islands = sorted_islands[-num_to_reset:]
            surviving_islands = sorted_islands[:-num_to_reset]

            print(
                f"Performing reset at generation {self.generation}. Resetting {num_to_reset} islands."
            )

            for reset_island in worst_islands:
                # Choose a surviving island uniformly at random
                if not surviving_islands:
                    print("No surviving islands to select from during reset.")
                    continue
                selected_island = random.choice(surviving_islands)

                # Retrieve the best program from the selected island
                best_program = selected_island.best_program
                if best_program is None:
                    print(
                        f"Selected island {selected_island.island_id} has no best program."
                    )
                    continue

                # Reset the island with the selected best program
                reset_island.programs = [best_program]
                reset_island.best_program = best_program
                print(
                    f"Island {reset_island.island_id} reset with program ID {best_program.program_id}, Score: {best_program.score}"
                )

    def _generate_and_evaluate_candidate(
            self, prompt: str, parents: List[Program], island_id: int
    ) -> Tuple[Program, int]:
        """Generate and evaluate a single candidate program"""
        try:
            competition = self.llm_pipeline.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=1.0,
                max_tokens=1000,
            )
            response = competition.choices[0].message.content
            with self.lock:
                self.total_price += (
                        competition.usage.prompt_tokens / 1_000_000 * self.completion_price
                        + competition.usage.completion_tokens
                        / 1_000_000
                        * self.prompt_price
                )

            if "```python" in response and "```" in response.split("```python")[1]:
                code = response.split("```python")[1].split("```")[0].strip()
            else:
                return None

            score = self._evaluate_program(code)

            if score is not None:
                return (
                    Program(
                        code=code,
                        score=score,
                        generation=self.generation,
                        parent_ids=[p.program_id for p in parents],
                        program_id=f"gen{self.generation}_{len(self.program_pool)}",
                    ),
                    island_id,
                )
            else:
                return None
        except Exception as e:
            print(f"Generation error: {e}")
            return None, island_id


# Usage example:
def main():
    # Load data
    df = pd.read_csv("processed_data.csv")

    X = df.drop(columns=["score"])
    y = df["score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Define evaluation function
    def evaluate(formula_func):
        return mean_absolute_error(y_test, formula_func(X_test))

    seed_program = """
def formula(df):
    return df["age"]
"""

    # Initialize FunSearch
    funsearch = FunSearch(
        eval_function=evaluate,
        llm_pipeline=client,
        islands=10,
        initial_program=seed_program,
        diversity_threshold=0.3,
        parallel_workers=4,
        price_limit=2,
        model="openai/gpt-4o-mini",
    )

    # Run evolution
    funsearch.evolve()


if __name__ == "__main__":
    main()
