#!/usr/bin/env node

/**
 * PugBrain MCP Server Wrapper
 *
 * This script spawns the Python pug-mcp server for use with MCP clients
 * like Claude Code, Cursor, Windsurf, and OpenClaw.
 */

const { spawn } = require('child_process');
const path = require('path');

// Find Python executable
function findPython() {
  const pythonCommands = ['python3', 'python', 'py -3'];

  for (const cmd of pythonCommands) {
    try {
      const result = require('child_process').spawnSync(
        cmd === 'py -3' ? 'py' : cmd,
        cmd === 'py -3' ? ['-3', '--version'] : ['--version'],
        { encoding: 'utf8', shell: true }
      );

      if (result.status === 0) {
        const version = result.stdout || result.stderr || '';
        const match = version.match(/Python (\d+)\.(\d+)/);
        if (match && parseInt(match[1]) >= 3 && parseInt(match[2]) >= 11) {
          return cmd === 'py -3' ? ['py', ['-3']] : [cmd, []];
        }
      }
    } catch {
      // Continue to next option
    }
  }

  return null;
}

// Main execution
function main() {
  const pythonInfo = findPython();

  if (!pythonInfo) {
    console.error('Error: Python 3.11+ is required but not found.');
    console.error('Please install Python 3.11 or later: https://www.python.org/downloads/');
    process.exit(1);
  }

  const [pythonCmd, pythonArgs] = pythonInfo;
  const args = [...pythonArgs, '-m', 'neural_memory.mcp'];

  // Spawn the Python MCP server
  const mcpServer = spawn(pythonCmd, args, {
    stdio: 'inherit',
    shell: true,
    env: { ...process.env }
  });

  mcpServer.on('error', (err) => {
    console.error('Failed to start pug-mcp:', err.message);
    console.error('Make sure pug-brain is installed: pip install pug-brain');
    process.exit(1);
  });

  mcpServer.on('exit', (code) => {
    process.exit(code || 0);
  });

  // Handle termination signals
  process.on('SIGINT', () => mcpServer.kill('SIGINT'));
  process.on('SIGTERM', () => mcpServer.kill('SIGTERM'));
}

main();
