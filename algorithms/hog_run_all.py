import subprocess

def run_python_script(script_path, *args):
    try:
        command = ['python3', script_path] + list(args)

        result = subprocess.run(command, capture_output=True, text=True)

        print(f"Output z {script_path}:")
        print(result.stdout)
        
        if result.stderr:
            print(f"Error z {script_path}:")
            print(result.stderr)

    except Exception as e:
        print(f"Błąd podczas uruchamiania {script_path}: {e}")

run_python_script('algorithms/hog.py', 'images_brugia')
run_python_script('algorithms/hog.py', 'cycle_gan_images_brugia')
run_python_script('algorithms/hog.py', 'sid_images_brugia')
run_python_script('algorithms/hog.py', 'custom_filter_images_brugia')

run_python_script('algorithms/hog.py', 'images_caorle')
run_python_script('algorithms/hog.py', 'cycle_gan_images_caorle')
run_python_script('algorithms/hog.py', 'sid_images_caorle')
run_python_script('algorithms/hog.py', 'custom_filter_images_caorle')

run_python_script('algorithms/hog.py', 'images_hungary_papa')
run_python_script('algorithms/hog.py', 'cycle_gan_images_hungary_papa')
run_python_script('algorithms/hog.py', 'sid_images_hungary_papa')
run_python_script('algorithms/hog.py', 'custom_filter_images_hungary_papa')

run_python_script('algorithms/hog.py', 'images_iseo_garibaldi')
run_python_script('algorithms/hog.py', 'cycle_gan_images_iseo_garibaldi')
run_python_script('algorithms/hog.py', 'sid_images_iseo_garibaldi')
run_python_script('algorithms/hog.py', 'custom_filter_images_iseo_garibaldi')