import sys
import subprocess

def main():
    args = sys.argv[1:]
    
    if args and args[0] == "--aggregate":
        # Run aggregator
        cmd = ["python3", "aggregate.py"] + args[1:]
        print(f"[entrypoint] Starting aggregator proxy: {cmd}")
    else:
        # Run single server
        cmd = ["python3", "server.py"] + args
        print(f"[entrypoint] Starting single model server: {cmd}")
        
    try:
        # Run the command and propagate return code
        res = subprocess.run(cmd)
        sys.exit(res.returncode)
    except KeyboardInterrupt:
        print("[entrypoint] Process interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        print(f"[entrypoint] Error launching command: {exc}")
        sys.exit(1)

if __name__ == "__main__":
    main()
