
def run_train(params: dict[str, int | float | str | dict[str, int | float | str]]):
    for epoch in range(1, params['epochs'] + 1):
        print(f"Epoch {epoch}/{params['epochs']} completed.")
