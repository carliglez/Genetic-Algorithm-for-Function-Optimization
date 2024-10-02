# Modifications of real code genetic algorithm for global optimization

## Overview

This program implements a Genetic Algorithm (GA) in Python for the task of locating the global minimum of multidimensional functions. The GA mimics the process of natural selection, evolving a population of potential solutions over multiple generations.

The program is organized into three folders:

- ``singlerun/``: Contains different versions of the GA and executes each of the implemented functions once.
- ``singlerun_time/``: Contains different versions of the GA and executes each of the implemented functions once, displaying the total execution time upon completion.
- ``hundredruns/``: Contains different versions of the GA and executes each of the implemented functions a total of 100 times.

The various versions of the GA are as follows:
- **GEN:** Base version of the GA.
- **GEN_S:** Implementing a new stopping rule.
- **GEN_S_M:** Incorporating a new stopping rule and a novel mutation mechanism.
- **GEN_S_M_LS:** Implementing a new stopping rule, a novel mutation mechanism, and the application of local search.

### Benchmark Functions

The following benchmark functions are included for optimization:
- Alluffi-Pentiny: ``alluffi_pentiny``
- Bohachevsky 1: ``bohachevsky_1``
- Bohachevsky 2: ``bohachevsky_2``
- Becker and Lago: ``becker_lago``
- Branin: ``branin``
- Camel: ``camel``
- Cb3 (Three-Hump Camel): ``cb3``
- Cosine Mixture (CM): ``cosine_mixture``
- DeJong: ``dejong``
- Easom: ``easom``
- Exponential: ``exponential``
- GKLS: ``gkls``
- Goldstein and Price: ``goldstein_price``
- Griewank2: ``griewank2``
- Hansen: ``hansen``
- Hartman 3: ``hartman_3``
- Hartman 6: ``hartman_6``
- Rastrigin: ``rastrigin``
- Rosenbrock: ``rosenbrock``
- Shekel 5: ``shekel_5``
- Shekel 7: ``shekel_7``
- Shekel 10: ``shekel10``
- Shubert: ``shubert``
- Sinusoidal: ``sinusoidal``
- Test2N: ``test2n``
- Test30N: ``test30n``
- Potential: ``potential``

### Genetic Algorithm Parameters

- Population Size (N): ``100``
- Maximum Iterations: ``200``
- Mutation Probability (``p_m``): ``0.05``
- User-defined Parameter for Mutation (``b``): ``5``
- Termination Criterion (``e``): ``1e-4``

### Execution Time

The execution time for running the genetic algorithm on all benchmark functions was recorded on a standard laptop. After running each version of the code three times, the recorded execution times are as follows:

- **GEN average execution time:** ``54.21 seconds``
- **GEN_S average execution time:** ``60.65 seconds``
- **GEN_S_M average execution time:** ``72.62 seconds``
- **GEN_S_M_LS average execution time:** ``49.61 seconds``

## Key Observations

- **GEN_S_M_LS** consistently has the lowest execution times among all configurations.
- **GEN** has the lowest execution time among its configurations, followed by **GEN_S** and **GEN_S_M**.
- **GEN_S_M** has the highest execution times among all configurations.

## Conclusion

The genetic algorithm effectively optimizes a diverse set of benchmark functions, demonstrating its capability to identify optimal solutions across different function types.

## Acknowledgments

This program is based on the article [Modifications of Real Code Genetic Algorithm for Global Optimization](https://www.sciencedirect.com/science/article/abs/pii/S0096300308002907), by Ioannis G. Tsoulos.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
