#!/usr/bin/env python3

import os
from os.path import dirname
from os.path import join
import time
import sys

import argparse
import configparser
from pprint import pprint
import subprocess
import multiprocessing
import json
import datetime

def parse_command_line():
    parser = argparse.ArgumentParser(description="Lunching application")
    parser.add_argument('-c', '--config_file', action='store')
    parser.add_argument('-m', '--machine_file', action='store')
    parser.add_argument('-p', '--profile', action='store')
    parser.add_argument('-w', '--work_path', action='store')
    args = parser.parse_args()

    return args

def parse_config_file (config_file_path, config):
    fconfig = configparser.ConfigParser()
    fconfig.read(config_file_path)
    for key, value in fconfig.items():
        if key not in config.keys():
            continue
        for ckey, cvalue in value.items():
                config[key][ckey] = cvalue
    return config

def get_default_config():
    config = { }
    config['application'] = {
    }
    config['worker'] = {
        'port' : '11000',
        'num_workers' : None,
        'num_gpus_per_worker' : '1',
        'sync' : 'true'

    }
    config['ps'] = {
        'port' : '12000',
        'num_ps' : None,
        'num_gpus_per_ps' : '1',
    }
    config['log'] = {
        'log_dir' : '/tmp',
        'logtofile' : 'true',
        'alsologtostderr' : 'false',
    }
    config['memory'] = {
        'mem_logger_dir' : None
    }
    config['strace'] = {
        'output' : '/tmp/master.strace',
        'summary' : 'false',
        'trace_set' : ''
    }
    config['valgrind'] = {
        'leak-check' : 'yes',
        'track-origins' : 'yes',
        'callgrind' : 'false'
    }
    config['hdfs'] = {
        'name_node' : 'hdfs://localhost:9000',
        'hadoop_classpath_file' : None
    }
    config['googleprof'] = {
        'output_dir' : '/tmp/tf.prof'
    }
    return config

def get_env_str(pargs, tf_config_str, num_gpus):
    env_vars = {
        'TF_CONFIG' : tf_config_str,
    }

    if pargs['hdfs']['hadoop_classpath_file'] is not None:
        with open(pargs['hdfs']['hadoop_classpath_file'], 'r') as fobj:
            env_vars['CLASSPATH'] = fobj.read().strip()

    if pargs['memory']['mem_logger_dir'] is not None:
        env_vars['TF_MEM_LOGGER_PATH_PREFIX'] = pargs['memory']['mem_logger_dir']

    if num_gpus == 0:
        env_vars['CUDA_VISIBLE_DEVICES'] = ""

    return ''.join([' %s=\'%s\'' % (k, v) for (k, v) in env_vars.items()])

def get_tf_config_dict(args, pargs):
    hosts = []
    with open(args.machine_file, 'r') as fobj:
        for line in fobj:
            hosts.append(line.strip())
    if not pargs['worker']['num_workers']:
         pargs['worker']['num_workers'] = str(len(hosts))
    else:
        pargs['worker']['num_workers'] = pargs['worker']['num_workers']
    if not pargs['ps']['num_ps']:
         pargs['ps']['num_ps'] = str(len(hosts))
    else:
         pargs['ps']['num_ps'] = pargs['ps']['num_ps']

    num_workers = int(pargs['worker']['num_workers'])
    num_ps = int(pargs['ps']['num_ps'])
    worker_hosts = hosts[0:num_workers]
    #ps_hosts = [host for host in hosts[num_workers:(num_ps + num_workers)]]
    ps_hosts = hosts[0:num_ps]

    worker_addrs = ["%s:%s" % (host, pargs['worker']['port']) for host in worker_hosts]
    ps_addrs = ["%s:%s" % (host, pargs['ps']['port']) for host in ps_hosts]
    cluster_spec = {
        "ps" : ps_addrs,
        "master" : worker_addrs
    }

    tf_config_dict = {
        "cluster" : cluster_spec,
        "environment" : "cloud"
    }
    return tf_config_dict, worker_hosts, ps_hosts

def get_logging_str(pargs):
    filename = 'tensorflow' + '.' + pargs['application']['model']
    time_str = datetime.datetime.now().strftime('%m%d-%H%M%S')
    filename += '.-%s-.' + time_str + '.%s'
    if pargs['log']['logtofile'] in ['True', 'true']:
        if pargs['log']['alsologtostderr'] in ['True', 'true']:
            logging_str = ' 2> >(tee ' + pargs['log']['log_dir'] + '/' + filename + ')'
        else:
            logging_str = ' 2> ' + pargs['log']['log_dir'] + '/' + filename
    else:
        logging_str = ''
    return logging_str

def get_strace_arg_str(strace_config, keyword, index):
    if strace_config['summary'] == 'true':
        arg_str = '-c'
    else:
        arg_str = '-tt -T -o ' + strace_config['output'] + '.' + keyword + '.' + str(index)

    if not strace_config['trace_set'] == '':
        arg_str += ' -e trace=' + strace_config['trace_set']

    return arg_str

def get_valgrind_arg_str(valgrind_config):
    if valgrind_config['callgrind'] == 'true':
        arg_str = '--tool=callgrind'
        return arg_str
    arg_str = ''.join([' --%s=%s' % (k, v) for (k, v) in valgrind_config.items() \
                       if not k == 'callgrind'])
    return arg_str

def get_googleprof_arg_str(googleprof_config, keyword, index):
    arg_str = ' LD_PRELOAD=' + googleprof_config['profiler_lib'] \
              + ' CPUPROFILE=' + googleprof_config['output_dir'] + '.' + keyword + '.' + str(index)
    return arg_str

def get_app_args_str(pargs_app):
    return ''.join([' --%s=%s' % (k, v) for (k, v) in pargs_app.items()])

def get_worker_args_str(pargs, worker_index, host_ip):
    worker_args_dict = {
        'master' : 'grpc://' + host_ip + ':' + pargs['worker']['port'],
        'worker_replicas' : pargs['worker']['num_workers'],
        'worker_gpu' : pargs['worker']['num_gpus_per_worker'],
        'worker_id' : str(worker_index),
        'worker_job' : '/job:master',
        'ps_replicas' : pargs['ps']['num_ps'],
        'ps_gpu' : pargs['ps']['num_gpus_per_ps'],
        'schedule' : 'train',
    }

    worker_args_str = ''.join([' --%s=%s' % (k, v) for (k, v) in worker_args_dict.items()])
    if pargs['worker']['sync'] == 'true':
        worker_args_str += ' --sync'
    return worker_args_str

def get_ps_args_str(pargs, ps_index, host_ip):
    ps_args_dict = {
        'master' : 'grpc://' + host_ip + ':' + pargs['ps']['port'],
        'schedule' : 'run_std_server',
        'ps_replicas' : pargs['ps']['num_ps'],
    }

    ps_args_str = ''.join([' --%s=%s' % (k, v) for (k, v) in ps_args_dict.items()])
    return ps_args_str

def get_command_with_profile(args, pargs, env_vars_str, job_name, index):
    trainer_path = 'python -u ./tensor2tensor/bin/t2t-trainer'

    if args.profile is None:
        cmd_app = 'cd ' + args.work_path + '; ' + env_vars_str + ' ' + trainer_path
    elif args.profile == 'strace':
        strace_arg_str = get_strace_arg_str(pargs['strace'], job_name, index)
        cmd_app = 'cd ' + args.work_path + '; ' + env_vars_str + ' strace ' + strace_arg_str + ' ' + trainer_path
    elif args.profile == 'valgrind':
        valgrind_arg_str = get_valgrind_arg_str(pargs['valgrind'])
        cmd_app = 'cd ' + args.work_path + '; ' + env_vars_str + ' valgrind ' + valgrind_arg_str + ' ' + trainer_path
    elif args.profile == 'googleprof':
        googleprof_arg_str = get_googleprof_arg_str(pargs['googleprof'], job_name, index)
        cmd_app = 'cd ' + args.work_path + '; ' + env_vars_str + ' ' + trainer_path
    else:
        print ('unsupported profile option %s' % args.profile)
        sys.exit(1)

    return cmd_app

if __name__ == '__main__':
    args = parse_command_line()
    pargs = get_default_config()
    assert args.config_file is not None
    pargs = parse_config_file(args.config_file, pargs)
    tf_config_dict, worker_hosts, ps_hosts = get_tf_config_dict(args, pargs)
    app_args_str = get_app_args_str(pargs['application'])
    print(app_args_str)
    wait_proc = None

    num_workers = int(pargs['worker']['num_workers'])
    num_ps = int(pargs['ps']['num_ps'])
    logging_str = get_logging_str(pargs)
    for worker_index in range(0, num_workers):
        print(worker_hosts[worker_index])
        host = worker_hosts[worker_index]
        tf_config_dict["task"] = {"type" : "master", "index" : worker_index}
        print(tf_config_dict)
        tf_config_str = json.dumps(tf_config_dict)
        print(tf_config_str)
        env_vars_str = get_env_str(pargs, tf_config_str, int(pargs['worker']['num_gpus_per_worker']))
        worker_args_str = get_worker_args_str(pargs, worker_index, host)
        cmd_worker = get_command_with_profile(args, pargs, env_vars_str, 'worker', worker_index) \
                     + app_args_str + worker_args_str + (logging_str % (host, 'worker' + str(worker_index)))
        print(cmd_worker)
        print(worker_args_str)
        worker_proc = subprocess.Popen(['ssh', '-oStrictHostKeyChecking=no',
                                        '-oUserKnownHostsFile=/dev/null',
                                        '-oLogLevel=QUIET',
                                        '%s' % host,
                                        cmd_worker],
                                       shell=False)
        if not wait_proc:
            wait_proc = worker_proc

    for ps_index in range(0, num_ps):
        host = ps_hosts[ps_index ]
        tf_config_dict["task"] = {"type" : "ps", "index" : ps_index}
        tf_config_str = json.dumps(tf_config_dict)
        print(tf_config_str)
        env_vars_str = get_env_str(pargs, tf_config_str, int(pargs['ps']['num_gpus_per_ps']))
        ps_args_str = get_ps_args_str(pargs, ps_index, host)
        cmd_ps = get_command_with_profile(args, pargs, env_vars_str, 'ps', ps_index) \
                 + app_args_str + ps_args_str + (logging_str % (host, 'ps' + str(ps_index)))
        print(cmd_ps)
        print(ps_args_str)
        ps_proc = subprocess.Popen(['ssh', '-oStrictHostKeyChecking=no',
                                    '-oUserKnownHostsFile=/dev/null',
                                    '-oLogLevel=QUIET',
                                    '%s' % host,
                                    cmd_ps],
                                   shell=False)

    wait_proc.wait(
)
