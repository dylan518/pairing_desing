from __future__ import annotations

from typing import Callable, Dict

import numpy as np


class ComprehensiveTestProblems:
    """Extended test suite with diverse problem characteristics."""

    @staticmethod
    def sphere(traits: np.ndarray) -> np.ndarray:
        """Sphere function: f(x) = sum(x²) - separable, unimodal."""
        return -np.sum(traits**2, axis=-1)

    @staticmethod
    def rosenbrock(traits: np.ndarray) -> np.ndarray:
        """Rosenbrock function: f(x) = sum(100(x_{i+1} - x_i²)² + (1-x_i)²) - non-separable, multimodal."""
        result = np.zeros(len(traits))
        for i in range(len(traits)):
            x = traits[i]
            sum_val = 0
            for j in range(len(x) - 1):
                sum_val += 100 * (x[j+1] - x[j]**2)**2 + (1 - x[j])**2
            result[i] = -sum_val
        return result

    @staticmethod
    def rastrigin(traits: np.ndarray) -> np.ndarray:
        """Rastrigin function: f(x) = sum(x² - 10cos(2πx) + 10) - separable, multimodal."""
        result = np.zeros(len(traits))
        for i in range(len(traits)):
            x = traits[i]
            sum_val = 0
            for j in range(len(x)):
                sum_val += x[j]**2 - 10 * np.cos(2 * np.pi * x[j]) + 10
            result[i] = -sum_val
        return result

    @staticmethod
    def ackley(traits: np.ndarray) -> np.ndarray:
        """Ackley function: non-separable, multimodal."""
        result = np.zeros(len(traits))
        for i in range(len(traits)):
            x = traits[i]
            d = len(x)
            sum1 = np.sum(x**2)
            sum2 = np.sum(np.cos(2 * np.pi * x))
            result[i] = -(-20 * np.exp(-0.2 * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + 20 + np.e)
        return result

    @staticmethod
    def griewank(traits: np.ndarray) -> np.ndarray:
        """Griewank function: non-separable, multimodal."""
        result = np.zeros(len(traits))
        for i in range(len(traits)):
            x = traits[i]
            sum_val = np.sum(x**2 / 4000)
            prod_val = 1
            for j in range(len(x)):
                prod_val *= np.cos(x[j] / np.sqrt(j + 1))
            result[i] = -(1 + sum_val - prod_val)
        return result

    @staticmethod
    def schwefel(traits: np.ndarray) -> np.ndarray:
        """Schwefel function: separable, multimodal."""
        result = np.zeros(len(traits))
        for i in range(len(traits)):
            x = traits[i]
            d = len(x)
            sum_val = np.sum(x * np.sin(np.sqrt(np.abs(x))))
            result[i] = -(418.9829 * d - sum_val)
        return result

    @staticmethod
    def levy(traits: np.ndarray) -> np.ndarray:
        """Levy function: multimodal."""
        result = np.zeros(len(traits))
        for i in range(len(traits)):
            x = traits[i]
            w = 1 + (x - 1) / 4
            term1 = np.sin(np.pi * w[0])**2
            term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
            term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
            result[i] = -(term1 + term2 + term3)
        return result

    @staticmethod
    def michalewicz(traits: np.ndarray) -> np.ndarray:
        """Michalewicz function: multimodal, m=10."""
        result = np.zeros(len(traits))
        m = 10
        for i in range(len(traits)):
            x = traits[i]
            sum_val = 0
            for j in range(len(x)):
                sum_val += np.sin(x[j]) * np.sin((j + 1) * x[j]**2 / np.pi)**(2 * m)
            result[i] = sum_val
        return result

    @staticmethod
    def zakharov(traits: np.ndarray) -> np.ndarray:
        """Zakharov function: non-separable."""
        result = np.zeros(len(traits))
        for i in range(len(traits)):
            x = traits[i]
            sum1 = np.sum(x**2)
            sum2 = np.sum(0.5 * np.arange(1, len(x) + 1) * x)
            result[i] = -(sum1 + sum2**2 + sum2**4)
        return result

    @staticmethod
    def dixon_price(traits: np.ndarray) -> np.ndarray:
        """Dixon-Price function: unimodal."""
        result = np.zeros(len(traits))
        for i in range(len(traits)):
            x = traits[i]
            sum_val = (x[0] - 1)**2
            for j in range(1, len(x)):
                sum_val += (j + 1) * (2 * x[j]**2 - x[j-1])**2
            result[i] = -sum_val
        return result

    @staticmethod
    def powell(traits: np.ndarray) -> np.ndarray:
        """Powell function: non-separable."""
        result = np.zeros(len(traits))
        for i in range(len(traits)):
            x = traits[i]
            sum_val = 0
            for j in range(0, len(x) - 3, 4):
                if j + 3 < len(x):
                    sum_val += (x[j] + 10 * x[j+1])**2 + 5 * (x[j+2] - x[j+3])**2 + \
                              (x[j+1] - 2 * x[j+2])**4 + 10 * (x[j] - x[j+3])**4
            result[i] = -sum_val
        return result

    @staticmethod
    def sum_squares(traits: np.ndarray) -> np.ndarray:
        """Sum Squares function: separable, unimodal."""
        result = np.zeros(len(traits))
        for i in range(len(traits)):
            x = traits[i]
            sum_val = 0
            for j in range(len(x)):
                sum_val += (j + 1) * x[j]**2
            result[i] = -sum_val
        return result

    @staticmethod
    def trid(traits: np.ndarray) -> np.ndarray:
        """Trid function: non-separable."""
        result = np.zeros(len(traits))
        for i in range(len(traits)):
            x = traits[i]
            sum1 = np.sum((x - 1)**2)
            sum2 = 0
            for j in range(len(x) - 1):
                sum2 += x[j] * x[j+1]
            result[i] = -(sum1 - sum2)
        return result

    @staticmethod
    def perm(traits: np.ndarray) -> np.ndarray:
        """Perm function: multimodal."""
        result = np.zeros(len(traits))
        beta = 10
        for i in range(len(traits)):
            x = traits[i]
            sum_val = 0
            for j in range(len(x)):
                for k in range(len(x)):
                    sum_val += ((k + 1 + beta) * (x[k]**(j + 1) - 1 / (k + 1)**(j + 1)))**2
            result[i] = -sum_val
        return result

    @staticmethod
    def get_all_problems() -> Dict[str, dict]:
        """Get all test problems with their characteristics."""
        return {
            "Sphere": {"func": ComprehensiveTestProblems.sphere, "type": "separable", "modality": "unimodal"},
            "Rosenbrock": {"func": ComprehensiveTestProblems.rosenbrock, "type": "non-separable", "modality": "multimodal"},
            "Rastrigin": {"func": ComprehensiveTestProblems.rastrigin, "type": "separable", "modality": "multimodal"},
            "Ackley": {"func": ComprehensiveTestProblems.ackley, "type": "non-separable", "modality": "multimodal"},
            "Griewank": {"func": ComprehensiveTestProblems.griewank, "type": "non-separable", "modality": "multimodal"},
            "Schwefel": {"func": ComprehensiveTestProblems.schwefel, "type": "separable", "modality": "multimodal"},
            "Levy": {"func": ComprehensiveTestProblems.levy, "type": "non-separable", "modality": "multimodal"},
            "Michalewicz": {"func": ComprehensiveTestProblems.michalewicz, "type": "non-separable", "modality": "multimodal"},
            "Zakharov": {"func": ComprehensiveTestProblems.zakharov, "type": "non-separable", "modality": "multimodal"},
            "Dixon-Price": {"func": ComprehensiveTestProblems.dixon_price, "type": "non-separable", "modality": "unimodal"},
            "Powell": {"func": ComprehensiveTestProblems.powell, "type": "non-separable", "modality": "multimodal"},
            "Sum_Squares": {"func": ComprehensiveTestProblems.sum_squares, "type": "separable", "modality": "unimodal"},
            "Trid": {"func": ComprehensiveTestProblems.trid, "type": "non-separable", "modality": "unimodal"},
            "Perm": {"func": ComprehensiveTestProblems.perm, "type": "non-separable", "modality": "multimodal"},
        }


def make_problem(task_label: str, trait_dim: int = 48):
    problems = ComprehensiveTestProblems.get_all_problems()
    if task_label not in problems:
        raise ValueError(f"Unknown problem label: {task_label}")
    func = problems[task_label]["func"]
    def fitness_fn(traits: np.ndarray) -> np.ndarray:
        return np.asarray(func(traits), dtype=float)
    return trait_dim, fitness_fn





