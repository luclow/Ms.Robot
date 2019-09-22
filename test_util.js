// Execute all unit tests in the current directory. 
// Takes jasmine_util from tfjs-core.
// Use the tfjs-core module from the right test directory.

function runTests(jasmineUtil, specFiles) {
  // tslint:disable-next-line:no-require-imports
  const jasmineConstructor = require('jasmine');

  Error.stackTraceLimit = Infinity;

  process.on('unhandledRejection', e => {
    throw e;
  });

  jasmineUtil.setTestEnvs(
      [{name: 'node', factory: jasmineUtil.CPU_FACTORY, features: {}}]);

  const runner = new jasmineConstructor();
  runner.loadConfig({spec_files: specFiles, random: false});
  runner.execute();
}

module.exports = {runTests};
