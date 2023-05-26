from view.window import Window


class Main:
    def __init__(self) -> None:
        self.window = Window()
    
    def start_app(self):
        self.window.run_window()

if __name__ == "__main__":
    main = Main()
    main.start_app()