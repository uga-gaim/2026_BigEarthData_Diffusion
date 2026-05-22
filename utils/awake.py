import time

from yaspin import yaspin


def stay_awake():
    counter = 0
    
    try:
        with yaspin(text="Staying awake ...") as spinner:
            while True:
                time.sleep(60)
                counter += 1
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        print(f"Awake for {counter} min!")
