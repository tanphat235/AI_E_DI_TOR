"""The command line interface - AIVE's contract with the AI director.

Two rules govern everything in this package:

* stdout carries machine-readable output only. Logs and human-facing text go to
  stderr. See :mod:`app.cli.output`.
* Failures exit non-zero and print a structured error, so the agent can correct
  itself instead of guessing at a traceback.
"""
