from evidently.ui.workspace import Workspace


def workspace():
    WORKSPACE_PATH="evidently_workspace"
    ws=Workspace.create(WORKSPACE_PATH)

    print("Workspace created successfully!")


if __name__=="__main__":
    try:
        workspace()

    except Exception as e:
        print(f"Error: {e}")