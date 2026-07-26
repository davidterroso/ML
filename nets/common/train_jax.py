
def run_train(params: dict[str, float | str]):
    for epoch in range(1, params['epochs'] + 1):
        print(f"Epoch {epoch}/{params['epochs']} completed.")
    return
