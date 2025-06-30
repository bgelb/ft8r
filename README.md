# ft8r

This repository explores building an FT8 demodulator.  The current test
suite verifies that the WSJT-X utilities installed on the system can
correctly round-trip an FT8 message.

Run the tests with `pytest`:

```bash
pytest -q
```

The test uses `ft8sim` to create a `.wav` file and decodes it with `jt9`.
It verifies that the decoded message and reported SNR, frequency and
time offset closely match the values used to generate the sample.
