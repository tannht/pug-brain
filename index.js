/**
 * PugBrain - Hybrid Neural-Vector Intelligence Core
 *
 * This is an npm wrapper for the PugBrain MCP server.
 * The actual MCP server is implemented in Python.
 *
 * Usage:
 *   npm install -g pug-brain
 *   pug-mcp  # Starts the MCP server
 *
 * For MCP client configuration:
 *   {
 *     "mcpServers": {
 *       "pug-brain": {
 *         "command": "pug-mcp"
 *       }
 *     }
 *   }
 */

module.exports = {
  name: 'pug-brain',
  version: '2.27.1',
  description: 'MCP server for PugBrain - Hybrid Neural-Vector Intelligence Core'
};
