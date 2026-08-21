import matplotlib
matplotlib.use("Agg")  # non-interactive backend: avoids GUI-backend flakiness/failures
                        # (e.g. tkinter) when many figures accumulate across tests, and is
                        # required outright on headless CI runners with no display.
