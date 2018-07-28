# Improving Energy Efficiency of Field-Coupled Nanocomputing Circuits by Evolutionary Synthesis
DOI: (not yet assigned)

# Contact us
If you have:
- questions about this work
- suggestions of improvements
- collaboration ideas

contact me in my email [marcoantonio@dcc.ufmg.br](mailto:marcoantonio@dcc.ufmg.br).

# Cite us

```
@inproceedings{ribeiro2018improving1,
  author    = {M. A. Ribeiro and I. A. de Carvalho and J. F. Chaves and G. L. Pappa and O. P. Vilela Neto},
  booktitle = {2018 IEEE Congress on Evolutionary Computation (CEC)},
  title     = {Improving Energy Efficiency of Field-Coupled Nanocomputing Circuits by Evolutionary Synthesis},
  year      = {2018},
  month     = {July},
  note      = {forthcoming}
}
```

# Dependencies

Make sure to download the latest version (as of July, 2018) of all dependencies below.

## OS dependencies
- python3
- llvm-dev
- texlive-full

## Python 3 dependencies
- numpy
- numba
- matplotlib

# How to run

## Evolving
Call `$ darwin.py <input> -verilog <output>` to evolve the `<input>` and save it as a verilog on the `<output>`.
Both files should use the ABC [1] verilog AIG format extended by the MAJ gates in the format below
(we added some examples into the folder `data/benchmarks`).

There are many other command line options available. Those are explained in details by
calling `$ darwin.py -help`.

### MAJ format

Majority gates are supported in the format `assign n1 = (i1 & i2) | (i1 & i3) | (i2 & i3);`,
where `n1 = MAJ(i1, i2, i3)`.

# References
[1] Brayton, R. and Mishchenko, A., 2010, July. ABC: An academic industrial-strength verification tool.
In _International Conference on Computer Aided Verification_ (pp. 24-40). Springer, Berlin, Heidelberg.
