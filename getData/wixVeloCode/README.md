# Wix Velo Backend Code

This directory contains the custom JavaScript code designed to run on the **Wix website backend** using Velo.

## Purpose

The code implements the necessary HTTP functions and utility services to expose Wix data securely to external microservices.

- **`http-functions.js`**: Defines the public REST endpoints (`/yourEventsEndpoint`, `/yourPostsEndpoint`, etc.) that the FastAPI service calls.
- **`utils.web.js`**: Contains core logic, specifically the function to **validate and decode the incoming JWT token** and check the associated subject (`sub`) claim against a stored secret.
- **`services.web.js`**: Contains the business logic for querying the actual Wix database collections (Events, Blog, etc.) using Wix APIs.

**Configuration Note:** The JWT signing secret used for authentication must be configured in the Wix Secrets Manager.

## Documentation

[Wix Velo API](https://dev.wix.com/docs/velo)
