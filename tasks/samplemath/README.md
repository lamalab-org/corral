
```python
environments = {
    "math_1": MathEnvironment("math_1", "What is 23 + 45?", 68),
    "math_2": MathEnvironment("math_2", "What is 12 * 8?", 96),
}

app = create_benchmark_server(environments)
import uvicorn

uvicorn.run(app, host="0.0.0.0", port=8000)
```


### For env with io

- Export CORRAL_FS_PROTOCOL & BASE_IO_PATH
