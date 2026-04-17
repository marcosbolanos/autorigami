# Main instructions
- Instead of writing defensive code, ensure having very explicit type contracts so that the inputs of a function behave as expected
- Don't make any dependency fallbacks. Just make sure necessary packages are
  installed. Also don't write any code that checks if packages are installed,
  just install them
- Use `uv` as a package manager with `uv add` and use `uv run` to run commands
