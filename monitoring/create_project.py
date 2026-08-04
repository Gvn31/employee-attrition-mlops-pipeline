from evidently.ui.workspace import Workspace


def create_project():
    """Create an Evidently workspace Project for monitoring
    """

    WORKSPACE_PATH="evidently_workspace"
    ws=Workspace.create(WORKSPACE_PATH)

    project=ws.create_project(
        name="Employee Attrition Monitoring",
        description="Production montoring dashboard for Employee Attrition MLOps Pipeline",
    )

    print("Project created successfully!")
    print(f"Project ID: {project.id}")


if __name__ == "__main__":
    try:
        create_project()

    except Exception as e:
        print(f"Error: {e}")

