import sys
import json
import wandb

def convert_run_to_jsonl(run_path):
    """
    Fetches a wandb run and exports its full history to a JSONL file.
    
    Args:
        run_path (str): The wandb path in the format 'entity/project/run_id'
    """
    try:
        # Initialize the API
        api = wandb.Api()
        
        # Fetch the run
        print(f"Fetching run: {run_path}...")
        run = api.run(run_path)
        
        # Define output filename based on run ID
        output_filename = f"{run.id}.jsonl"
        
        # We use scan_history() to ensure we get all data points, 
        # not just the sampled history used for plotting.
        history = run.scan_history()
        
        print(f"Writing data to {output_filename}...")
        
        line_count = 0
        with open(output_filename, 'w') as f:
            for row in history:
                # wandb returns a dict-like object, convert to standard dict
                # We also merge the config/summary if you want context per line,
                # but standard practice is usually just the metrics per step.
                json_line = json.dumps(row)
                f.write(json_line + '\n')
                line_count += 1
                
        print(f"Success! Converted {line_count} steps to {output_filename}")

    except wandb.errors.CommError as e:
        print(f"Error: Could not connect to WandB. Check your API key. {e}")
    except ValueError:
        print("Error: Invalid run path. Format should be 'entity/project/run_id'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Check for arguments
    if len(sys.argv) < 2:
        print("Usage: python wandb_to_jsonl.py <entity/project/run_id>")
        sys.exit(1)

    run_path_arg = sys.argv[1]
    convert_run_to_jsonl(run_path_arg)
