# from i24_configparse               import parse_cfg
# from i24_logger.log_writer         import logger,connect_automatically,catch_critical
# import os
# import numpy 
# # load parameters
# params = parse_cfg("TRACK_CONFIG_SECTION",
#                     cfg_name="execute.config", SCHEMA=False)


# global logger

# # initialize logger
# log_params = {
#             "log_name":"Tracking Session",
#             "processing_environment":os.environ["TRACK_CONFIG_SECTION"],
#             "logstash_address":(params.log_host_ip,params.log_host_port),
#             "connect_logstash": (True if "logstash" in params.log_mode else False),
#             "connect_syslog":(True if "syslog" in params.log_mode else False),
#             "connect_file": (True if "file" in params.log_mode else False),
#             "connect_console":(True if "sysout" in params.log_mode else False),
#             "console_log_level":params.log_level
#             }

# connect_automatically(user_settings = log_params)
     
     
     