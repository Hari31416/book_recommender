import os, time


def embedding_dim_experiments():
    """Runs experiments for different embedding dimensions."""
    embedding_dims = [32, 40, 50, 64]
    run_names = [f"embedding_dim_{embedding_dim}" for embedding_dim in embedding_dims]
    for i in range(len(run_names)):
        run_name = run_names[i]
        embedding_dim = embedding_dims[i]
        args = [
            "python",
            "train.py",
            "--embedding_dim",
            str(embedding_dim),
            "--run_name",
            run_name,
        ]
        cmd = " ".join(args)
        print(f"WORKING ON {run_name}")
        os.system(cmd)
        time.sleep(15)
        print(f"FINISHED {run_name}")
        print("-----" * 10)


def negative_sample_ratio_experiments(debug=False):
    """Runs experiments for different embedding dimensions."""
    negative_ratios_dims = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    run_names = [
        f"negative_sample_ratio_{i+1}" for i in range(len(negative_ratios_dims))
    ]
    for i in range(len(run_names)):
        run_name = run_names[i]
        negative_sample_ratio = negative_ratios_dims[i]
        args = [
            "python",
            "train.py",
            "--negative_samples_ratio",
            str(negative_sample_ratio),
            "--run_name",
            run_name,
        ]
        cmd = " ".join(args)
        if debug:
            print(cmd)
            continue
        print(f"WORKING ON {run_name}")
        os.system(cmd)
        time.sleep(15)
        print(f"FINISHED {run_name}")
        print("-----" * 10)


def cf_layers_experiments(debug=False):
    """Runs experiments for different embedding dimensions."""
    cf_layers_dims = [
        [20, 20],
        [32, 32],
        [32, 16],
        [32, 16, 8],
        [32, 16, 16],
        [32, 16, 8, 8],
    ]
    run_names = [f"cf_layers_{i+1}" for i in range(len(cf_layers_dims))]
    for i in range(len(run_names)):
        if i < 2:
            continue
        run_name = run_names[i]
        cf_layers = cf_layers_dims[i]
        cf_layers = " ".join([str(x) for x in cf_layers])
        args = [
            "python",
            "train.py",
            "--cf_layer_neurons",
            cf_layers,
            "--run_name",
            run_name,
        ]
        cmd = " ".join(args)
        if debug:
            print(cmd)
            continue
        print(f"WORKING ON {run_name}")
        os.system(cmd)
        time.sleep(15)
        print(f"FINISHED {run_name}")
        print("-----" * 10)


if __name__ == "__main__":
    # embedding_dim_experiments()
    # negative_sample_ratio_experiments(debug=False)
    cf_layers_experiments(debug=False)
