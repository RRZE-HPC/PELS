
# PELS: Performance Engineering for (sparse) Linear Solvers Demo

This is the hands-on material for PELS tutorial.
To follow the tutorial, please open PELS.ipynb in Jupyter notebook and follow the tutorial there.

### Terminal version

If you prefer doing experiments from teh command-line, you can execute the Python files
``kernels.py`` and ``pcg.py``. The flag ``--help`` will describe how to run certain benchmarks and demos, e.g.:

```bash
python3 pcg.py --help
```

## Notes for developing/updating

- When updating the Python programs, try to keep the ``test_*.py`` files up to date
  and make sure that nothing breaks by regularly running ``pytest``.
- Before commiting changes to the Jupyter notebook ``PELS.ipynb``, set up your local repo to
  filter out changes in execution counts etc.:
  ```bash
  pip install pre-commit
  pre-commit install
  ```
