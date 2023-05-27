from view.view_main import ViewMain

class Main:
    def __init__(self) -> None:
        self.view_main = ViewMain()
    
    def start_app(self):
        self.view_main.run_window()

if __name__ == "__main__":
    main = Main()
    main.start_app()