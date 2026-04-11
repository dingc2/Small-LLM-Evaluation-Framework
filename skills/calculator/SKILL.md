# Calculator Skill

## Name
`calculator`

## Description
Evaluates arithmetic and mathematical expressions safely, without `eval`.
Supports basic operations (`+`, `-`, `*`, `/`, `**`, `%`), parentheses,
integer and floating-point literals, and common functions via the `math`
module (`sqrt`, `sin`, `cos`, `log`, `abs`, `ceil`, `floor`, `pi`, `e`).

## Trigger Patterns
Any query that contains:
- Keywords: `calculate`, `compute`, `evaluate`, `solve`, `math`, `what is`
- Arithmetic operators: `+`, `-`, `*`, `/`
- Numeric expressions: digits adjacent to operators

## Input
```json
{
  "query": "sqrt(144) + 2 ** 8"
}
```

## Output
```json
{
  "result": 268.0,
  "success": true,
  "metadata": {
    "expression": "sqrt(144) + 2 ** 8",
    "type": "float"
  }
}
```

## Error handling
- Division by zero → `SkillOutput(success=False, error="division by zero")`
- Unsupported function → descriptive error message
- Malformed expression → parse error message

## Notes
- Uses `ast.parse` + a whitelist node visitor; **no `eval`** is called.
- Numbers larger than 1e308 raise an overflow error.
- String / boolean inputs return an error rather than raising.
