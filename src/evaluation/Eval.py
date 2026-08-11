from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd


def load_lightning_logs(log_dir):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    data = {}

    for tag in event_acc.Tags()["scalars"]:
        events = event_acc.Scalars(tag)

        data[tag] = {
            "step": [e.step for e in events],
            "value": [e.value for e in events],
        }

    return data

# logs = load_lightning_logs("./lightning_logs/version_0")

# for name, values in logs.items():
#     print(name)
    print(values)

def logs_to_dataframe(log_dir):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    rows = []

    for tag in event_acc.Tags()["scalars"]:
        for event in event_acc.Scalars(tag):
            rows.append({
                "step": event.step,
                "metric": tag,
                "value": event.value,
            })

    return pd.DataFrame(rows)



for i in range(12):
    df = logs_to_dataframe(f"./lightning_logs/version_{i}")

    df.to_csv(f"./PCEL_assignment/src/results/res_{i}.csv")

