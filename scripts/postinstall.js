#!/usr/bin/env node

/**
 * PugBrain Post-Install Script
 *
 * Checks for Python 3.11+ and guides users to install the Python package
 * if not already installed.
 */

const { spawnSync } = require('child_process');
const https = require('https');

const CYAN = '\x1b[36m';
const GREEN = '\x1b[32m';
const YELLOW = '\x1b[33m';
const RED = '\x1b[31m';
const RESET = '\x1b[0m';
const BOLD = '\x1b[1m';

function log(color, message) {
  console.log(`${color}${message}${RESET}`);
}

function findPython() {
  const pythonCommands = ['python3', 'python', 'py -3'];

  for (const cmd of pythonCommands) {
    try {
      const result = spawnSync(
        cmd === 'py -3' ? 'py' : cmd,
        cmd === 'py -3' ? ['-3', '--version'] : ['--version'],
        { encoding: 'utf8', shell: true }
      );

      if (result.status === 0) {
        const version = result.stdout || result.stderr || '';
        const match = version.match(/Python (\d+)\.(\d+)/);
        if (match && parseInt(match[1]) >= 3 && parseInt(match[2]) >= 11) {
          return { cmd, version: `${match[1]}.${match[2]}` };
        }
      }
    } catch {
      // Continue to next option
    }
  }

  return null;
}

function checkPugBrainInstalled() {
  const result = spawnSync('pip', ['show', 'pug-brain'], {
    encoding: 'utf8',
    shell: true
  });

  return result.status === 0;
}

function main() {
  console.log('');
  log(CYAN, '🐶 PugBrain MCP Server - Post-Install Check');
  console.log('');

  // Check for Python
  const python = findPython();

  if (!python) {
    log(YELLOW, '⚠ Python 3.11+ not found');
    console.log('');
    log(RESET, 'To use PugBrain MCP, please install Python 3.11+:');
    log(CYAN, '  https://www.python.org/downloads/');
    console.log('');
    return;
  }

  log(GREEN, `✓ Python ${python.version} found (${python.cmd})`);

  // Check for pug-brain Python package
  if (checkPugBrainInstalled()) {
    log(GREEN, '✓ pug-brain Python package is installed');
    console.log('');
    log(CYAN, 'You can now use pug-mcp with your MCP client!');
    console.log('');
    log(RESET, 'For Claude Code, add to your MCP config:');
    console.log('  {');
    console.log('    "mcpServers": {');
    console.log('      "pug-brain": {');
    console.log('        "command": "pug-mcp"');
    console.log('      }');
    console.log('    }');
    console.log('  }');
    console.log('');
  } else {
    log(YELLOW, '⚠ pug-brain Python package not found');
    console.log('');
    log(RESET, 'To complete installation (MCP server only):');
    log(CYAN, '  pip install pug-brain');
    console.log('');
    log(RESET, 'For web dashboard + REST API:');
    log(CYAN, '  pip install "pug-brain[server]"');
    console.log('');
    log(RESET, 'Note: Quotes are required for [server] in zsh/bash');
    console.log('');
  }

  // Show quick links
  log(RESET, 'Documentation: https://github.com/tannht/pug-brain');
  console.log('');
}

main();
