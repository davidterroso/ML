from typing import Dict, Union

def run_train(params: Dict[str, Union[float, str]]):
    for epoch in range(1, params['epochs'] + 1):
        print(f"Epoch {epoch}/{params['epochs']} completed.")
    return