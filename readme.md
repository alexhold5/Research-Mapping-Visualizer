# Research Mapping Visualizer

## Running the visualizer locally

To run the website locally, perform the following steps:

- Clone the repository
- Install Node
- In a terminal, CD into the "web/" directory in the root
- Run the command "npm install"
- Run the command "npm run dev"
- Access the website at the localhost url provided in the terminal

## Accessing the current stable deployment

There is a github pages deployment of the website at the following url: https://alexhold5.github.io/Research-Mapping-Visualizer/. This URL is subject to change

## Deploying to the github page

To make changes to the website and deploy them, follow these steps:

- Create a branch off of "main" locally
- Edit and commit your changes, testing by running locally as you go
- Publish the branch
- In github, go to the "Actions" tab
- Select the "Deploy to GitHub Pages" workflow from the left sidebar
- Select "Run Workflow" in the top right, and select the branch you just published
- Ensure that the website deploys and runs correctly and bug-free at the URL before PRing or merging your changes into main. This is so that if anything breaks, we can rebuild the deployed website from the last stable branch at main
  - Note that in order to see your changes, you will likely need to click Ctrl + Shift + R on Chrome while in the tab in order to clear the previously cached view of the website. It may also take a minute or two to update after the workflow runs.
