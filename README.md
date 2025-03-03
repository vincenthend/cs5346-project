## Prerequisites

- [Node.js](https://nodejs.org/) (which includes npm)
- [pnpm](https://pnpm.io/) (install globally using `npm install -g pnpm`)

## Getting Started

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd dataviz-monorepo
   ```

2. **Install dependencies:**

   Run the following command in the root directory of the project:

   ```bash
   pnpm install
   ```

3. **Run the application:**

   Navigate to the `web` package and start the development server:

   ```bash
   cd packages/web
   pnpm dev
   ```

   The application will be available at `http://localhost:3000`.

## Scripts

- `dev`: Starts the development server.
- `build`: Builds the application for production.
- `lint`: Runs ESLint to check for code quality issues.

## ESLint Configuration

This project uses ESLint for linting JavaScript and TypeScript files. The configuration is set to enforce the following rules:

## License

This project is licensed under the MIT License. See the LICENSE file for details.