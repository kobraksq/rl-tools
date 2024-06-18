import {TrajectoryPlayer} from "./TrajectoryPlayer.js";
export class ShowRun{
    constructor(container, run){

        container.innerHTML = ""
        const step = Object.keys(run.steps).sort().reverse()[0]

        const description = document.createElement("div")

        const latest_run_experiment = document.createElement("span")
        latest_run_experiment.innerText = run.config.experiment
        const experiment_label = document.createElement("b")
        experiment_label.innerText = "Experiment: "
        description.appendChild(experiment_label)
        description.appendChild(latest_run_experiment)
        const latest_run_commit_hash = document.createElement("span")
        latest_run_commit_hash.innerText = run.config.commit
        const commit_label = document.createElement("b")
        description.appendChild(document.createElement("br"))
        commit_label.innerText = "Commit: "
        description.appendChild(commit_label)
        description.appendChild(latest_run_commit_hash)
        const latest_run_name = document.createElement("span")
        latest_run_name.innerText = run.config.name
        const name_label = document.createElement("b")
        description.appendChild(document.createElement("br"))
        name_label.innerText = "Name: "
        description.appendChild(name_label)
        description.appendChild(latest_run_name)
        const latest_run_config = document.createElement("span")
        latest_run_config.innerText = JSON.stringify(run.config.population)
        const config_label = document.createElement("b")
        description.appendChild(document.createElement("br"))
        config_label.innerText = "Config: "
        description.appendChild(config_label)
        description.appendChild(latest_run_config)
        const latest_run_seed = document.createElement("span")
        latest_run_seed.innerText = run.config.seed
        description.appendChild(document.createElement("br"))
        const seed_label = document.createElement("b")
        seed_label.innerText = "Seed: "
        description.appendChild(seed_label)
        description.appendChild(latest_run_seed)
        const latest_run_checkpoint = document.createElement("span")
        latest_run_checkpoint.innerText = parseInt(step).toString()
        description.appendChild(document.createElement("br"))
        description.appendChild(latest_run_checkpoint)

        container.appendChild(description)

        const trajectory_player_container = document.createElement("div")
        trajectory_player_container.style.height = "500px";
        const trajectory_player = new TrajectoryPlayer(run.ui_jsm);
        trajectory_player_container.appendChild(trajectory_player.getCanvas());
        container.appendChild(trajectory_player_container)
        trajectory_player.playTrajectories(run.steps[step].trajectories_compressed);

    }
}
