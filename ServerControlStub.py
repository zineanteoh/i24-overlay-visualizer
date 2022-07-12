import sys
import time
import configparser
import socket
import _pickle as pickle
import os
import sys
import json
import subprocess
import multiprocessing as mp
import signal

from i24_logger.log_writer import logger, catch_critical, log_errors


# def dummy_function(arg1,arg2,kwarg1 = None):
#     while True:
#         logger.debug("This is what happens when you get a zombie process!")
#         time.sleep(5)
        
        
        
# CODEWRITER TODO - import your process targets here such that these functions can be directly passed as the target to mp.Process or mp.Pool

# TODO - add your process targets to register_functions
# register_functions = [dummy_function]
# name_to_process = dict([(fn.__name__, fn) for fn in register_functions])


# CODEWRTIER TODO - Change the Name to <YOUR COMPONENT> Server Control"
logger.set_name("{} Server Control".format(str(socket.gethostname()).upper()))

# all processContainers have this format
processContainer_exampele = {
    "process": "mp.Process",
    "command": "name of target function",
    "timeout": 1,
    "args": [],
    "kwargs": {}, 
    "group": "INGEST" or "TRACKING" or "POSTPROCESSING" or "ARCHIVE",
    "description": "This process specifies 2 arguments, 0 keyword arguments and 0 flags at the Cluster Level. It expects 2 additional arguments and one additional keyword argument to be appended by ServerControl",
    "keep_alive":True
    }



class ServerControlStub:
    """
    ServerControl has a few main functions. 
    1. Continually open socket (server side) that listens for commands from ClusterControl
    2. Start and maintain a list of subprocesses and processes
    3. Monitor these processes and restart them as necessary
    4. Log any status changes
    """
    
    def __init__(self,name_to_process,sock_port = 5999):
        self.log_frequency = 30 # every _ seconds
        self.last_log = 0
        self.default_timeout = 5
        
        self.msg_to_fn = {
                         "CONFIG":self.configure,
                         "START":self.start,
                         "SOFT STOP":self.soft_stop,
                         "HARD STOP":self.hard_stop,
                         "STOP":self.soft_stop,
                         "FINISH":self.finish
                         } # which function to run for each TCP message
        
        self.name_to_process = name_to_process # which target function to use for string process name
        
        
        
        # create socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        server_address = (local_ip,sock_port)
        self.sock.bind(server_address)
        self.sock.listen(1)
        
        print("Waiting for client connection on: {}".format(server_address))
                
        # Wait for a connection
        try:
            self.connection, self.client_address = self.sock.accept()
            logger.debug("ServerControl connected to ClusterControl at {}".format(self.client_address))
        except: # I don't think this could ever really be triggered
            self.socket_cleanup()
        
        # to store processContainers
        self.process_list = []
        
        
        
        # CODEWRITER TODO - Implement any shared variables (queues etc. here)
        # each entry is a tuple (args,kwargs) (list,dict)
        # to be added to process args/kwargs and key is process_name
        self.additional_args = self.get_additional_args()
                                
        
        self.main()
       
        
    #%% MESSAGE HANDLER PROCESSES
    
    def configure(self,msg):
        # add each processContainer to self.process_list
        for pC in msg[1]:
            pC["keep_alive"] = False
            self.process_list.append(pC)
        logger.debug("Initialized {} processes".format(len(self.process_list)))
        
        
    def start(self,msg):
        group = msg[1]
        count = 0
        for pC in self.process_list:
            if group is None or pC["group"] == group:
                p_args = pC["args"]
                p_kwargs = pC["kwargs"]
                proc_name = pC["command"] 
                if proc_name in self.additional_args.keys():
                    p_args += self.additional_args[proc_name][0]
                    p_kwargs = {**p_kwargs, **self.additional_arg[proc_name][1]}
                
                p_target= self.name_to_process[proc_name]
                
                # create Process
                pid = mp.Process(target = p_target,args = p_args,kwargs = p_kwargs,daemon = bool(pC["daemon"]))
                
                # add Process to process_list
                pC["process"] = pid
                pC["process"].start()
                count += 1
                pC["keep_alive"] = True
        logger.debug("Started {} processes with group: {}".format(count,group))

    def soft_stop(self,msg):
        stop_pids = []
        to_stop = 0
        group = msg[1]
        for pC in self.process_list:
            if (group is None or pC["group"] == group) and pC["keep_alive"]:
                sig = signal.SIGINT
                stop_pids.append(pC["process"].pid)
                os.kill(pC["process"].pid,sig)
                pC["keep_alive"] = False
                to_stop += 1
                
        if to_stop > 0:
            logger.debug("Sent signal {} to {} processes with group: {}".format(sig,to_stop,group))
        
        # Give each process a change to soft stop before killing
        soft_stop_time = time.time()
        
        not_done = True
        while not_done:
            not_done = False
            for pC in self.process_list:
                if pC["process"].pid in stop_pids:
                    if pC["keep_alive"] and pC["process"].is_alive():
                        not_done = True
                        timeout = self.default_timeout
                        if "timeout" in pC.keys():
                            timeout = pC["timeout"]
                        if time.time() - soft_stop_time > timeout:
                            sig = signal.SIGKILL
                            os.kill(pC["process"].pid,sig)
                            logger.warning("Process {} soft stop timed out; sent signal {}".format(pC["command"],sig))
                       
                    else:
                        pC["keep_alive"] = False

            
        logger.debug("Stopped {} processes with group: {}".format(to_stop,group))
        
        

        
    def hard_stop(self,msg):
        group = msg[1]
        count = 0
        for pC in self.process_list:
            if (group is None or pC["group"] == group) and pC["keep_alive"]:
                sig = signal.SIGKILL
                os.kill(pC["process"].pid,sig)
                count += 1
                pC["keep_alive"] = False
        if count > 0:
            logger.debug("Sent signal {} to {} processes with group: {}".format(sig,count,group))
        
    def finish(self,msg):
        group = msg[1]
        count = 0
        for pC in self.process_list:
            if (group is None or pC["group"] == group) and pC["keep_alive"]:
                sig = signal.SIGUSR1
                os.kill(pC["process"].pid,sig)
                count += 1
                pC["keep_alive"] = False
        if count > 0:
            logger.debug("Sent signal {} to {} processes with group: {}".format(sig,count,group))
        
        
    
    #%% Assorted OTHER FUNCTIONS
    def recv_msg(self,timeout = 0.01):
        self.connection.settimeout(timeout)

        try:
            payload = self.connection.recv(4096)
            return pickle.loads(payload)
        except socket.timeout:
            return None
        
    def socket_cleanup(self): 
        """ Close socket, log shutdown, etc"""
        logger.debug("Cleaning up sockets")
        self.sock.shutdown(socket.SHUT_RDWR)
        self.sock.close()
        self.connection.close()
        
    def keep_processes_alive(self): 
        for idx,pC in enumerate(self.process_list):
            if pC["keep_alive"] and not pC["process"].is_alive():
                
                # log dead process to logger
                logger.warning("Process {} (PID {}) died and is being restarted.".format(pC["command"],pC["process"].pid))
                
                # let's make sure it's really dead, we don't want any zombies around these parts
                pC["process"].kill()
                pC["process"].join()
                
                # restart a new Process with the same command
                self.restart_one(idx)
                
    def restart_one(self,idx):
        pC = self.process_list[idx]
        
        p_args = pC["args"]
        p_kwargs = pC["kwargs"]
        proc_name = pC["command"] 
        if proc_name in self.additional_args.keys():
            p_args += self.additional_args[proc_name][0]
            p_kwargs = {**p_kwargs, **self.additional_arg[proc_name][1]}
        
        p_target= self.name_to_process(proc_name)
        
        # create Process
        pid = mp.Process(target = p_target,args = p_args,kwargs = p_kwargs)
        
        # add Process to process_list
        pC["process"] = pid
        self.process_list[idx] = pC
        self.process_list[idx].start()
        
        logger.debug("Restarted {} process.".format(pC["command"]))
        
    def log_status(self):
        n_live = 0
        to_log = {}
        for pC in self.process_list:
            if "process" in pC.keys() and pC["process"].is_alive():
                n_live += 1
                to_log["PID:" + str(pC["process"].pid)] = [pC["command"],pC["group"],pC["process"].pid]
                
        logger.info("{} live processes".format(n_live),extra = to_log)
        self.last_log = time.time()
    
    #%% MAIN LOOP
    @catch_critical()   
    def main(self):
        
        while True:
            
            if time.time() - self.last_log > self.log_frequency:
                self.log_status()
            
            try:
                msg = self.recv_msg()
                # take action on relevant group
                if msg is not None:
                    command = msg[0]
                    self.msg_to_fn[command](msg)
                
                # regular actions
                self.keep_processes_alive()
        
            except Exception as e:
                if False and type(e).__name__ == "EOFError": # occurs sometimes when socket has no message?
                    print("Caught EOFERROR")
                    continue
                else:
                    self.socket_cleanup()
                    raise e
                    break



