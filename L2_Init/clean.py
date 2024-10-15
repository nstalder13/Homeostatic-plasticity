import subprocess
import time

def run_wandb_sync():
    # Replace 'N' with the actual number of hours you want to clean
    command = ["wandb", "sync", "--clean", "--clean-old-hours", "3"]
    try:
        # Use Popen to send input to the command
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Automatically send 'yes' to any prompts
        stdout, stderr = process.communicate(input=b"y\n")
        
        if process.returncode == 0:
            print("Command executed successfully.")
            print(stdout.decode())
        else:
            print(f"Error executing command: {stderr.decode()}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def main():
    while True:
        run_wandb_sync()
        # Wait for 3 hours (3 * 60 * 60 seconds)
        time.sleep(3 * 60 * 60)

if __name__ == "__main__":
    main()
