import numpy as np

# class TimeSync:
#     def __init__(self) -> None:

#         self.msg_queue = []         # append message of specific sensor to queue
#         self.ts_queue = []
#         self.last_pose_key = None

#     def add_to_queue(self, msg):
#         self.msg_queue.append(msg)

#     def add_to_ts_queue(self, ts):
#         self.ts_queue.append(ts)

#     def get_time_match(self, query_timestamps, target_timestamps):

#         query_timestamps = list(query_timestamps)

#         # If measurement in queue and the oldest measurment is later than current posekey
#         # Process the oldest measurements in self.msg_queue to decide if they are relevant for the current new_key
        
#         matched_timestamps = {}

#         last_pose_key = None
#         for new_key in range(1, len(target_timestamps)):
#             # print(new_key)
#             in_future = False
#             curr_time = target_timestamps[new_key] 
#             while(len(query_timestamps) > 1 and in_future is False):      

#                 oldest_measurement_time = query_timestamps[0] #(self.msg_queue[0].header.stamp.secs * 1_000_000_000 + self.msg_queue[0].header.stamp.nsecs) 
#                 next_measurement_time = query_timestamps[1] # (self.msg_queue[1].header.stamp.secs * 1_000_000_000 + self.msg_queue[1].header.stamp.nsecs) 
#                 # print("New key", new_key, "Last key", last_pose_key)
#                 # print("Kiba Curr", curr_time, "Oldest", oldest_measurement_time, "Next", next_measurement_time)

#                 if(oldest_measurement_time < curr_time):
#                     # print('oldest measurement time < curr time')
                    
#                     newer_key_time = target_timestamps[int(new_key)] 
#                     older_key_time = target_timestamps[int(new_key - 1)]
#                     time_to_current = abs(newer_key_time - oldest_measurement_time) 
#                     time_to_previous = abs(older_key_time - oldest_measurement_time) 

                

#                     if(time_to_current > time_to_previous):
#                         # Meas in queue better suited to previous key, keep working on prev key
#                         new_key -= 1
#                         if(new_key == last_pose_key):
#                             # print("popped 1")
#                             query_timestamps.pop(0)
#                         # else:
#                             # print("not popped yet 1")
#                     else:
#                         time_old_to_pose = abs(oldest_measurement_time - newer_key_time)
#                         time_next_to_pose = abs(next_measurement_time - newer_key_time)

#                         # Take care of where next measurement is not past next node
#                         if(next_measurement_time < newer_key_time):
#                             # print("popped 2")
#                             query_timestamps.pop(0)
                        
#                         # Take care of case where next measurement is better
#                         elif(time_old_to_pose > time_next_to_pose):
#                             # print("popped 3")
#                             query_timestamps.pop(0)
#                             # print("next meas better")

#                         else:
#                             # print("popped 4")
#                             # Actually add the factor
#                             query_stamp = query_timestamps.pop(0)
#                             matched_timestamps[target_timestamps[new_key]] = query_stamp                    
#                             last_pose_key = new_key

#                 else:
#                     in_future = True
#                     # print("In future")


#             # if new_key==5:
#             #     exit()

#         return matched_timestamps

#     def get_ts_match(self, new_key, timestamps):
#         ''''
#         new_key = int, The agent posekey id that you will start searching at
#         poseKey_to_time = dict being tracked by factor graph. keys: poseKey values: corresponding timestamp
#         '''

#         if len(timestamps) < 2:
#             return None, None

#         # A flag to determine if all remaining measurements are in the future relative to the current pose key's time
#         in_future = False

#         # new_key = len(timestamps) - 1
#         print("key", new_key)
#         curr_time = timestamps[new_key]
#         poseKey_to_add = None
#         ts_to_match = None

#         # If measurement in queue and the oldest measurment is later than current posekey
#         # Process the oldest measurements in self.msg_queue to decide if they are relevant for the current new_key
#         while(len(self.ts_queue) > 1 and in_future is False):  

#             oldest_measurement_time = (self.ts_queue[0]) # nanosecs 
#             next_measurement_time = (self.ts_queue[1]) # nanosecs 
#             # print("Curr", curr_time, "Oldest", oldest_measurement_time, "Next", next_measurement_time)

#             if(oldest_measurement_time < curr_time):
#                 # print('oldest measurement time < curr time')
                
#                 newer_key_time = timestamps[int(new_key)] 
#                 # print("newer key time: %d\n"%newer_key_time)
#                 older_key_time = timestamps[int(new_key - 1)]
#                 # print("older key time: %d\n"% older_key_time)
#                 time_to_current = abs(newer_key_time - oldest_measurement_time) 
#                 # print("time to current: %d\n"%time_to_current)
#                 time_to_previous = abs(older_key_time - oldest_measurement_time) 
#                 # print("time to previous: %d\n"%time_to_previous)

#                 if(time_to_current > time_to_previous):
#                     # Meas in queue better suited to previous key, keep working on prev key
#                     # print('time to current is longer than time to prev')
#                     new_key -= 1
#                     if(new_key == self.last_pose_key):
#                         print("popped 1")
#                         self.ts_queue.pop(0)
#                 else:
#                     # print('time to current is shortest')

#                     time_old_to_pose = abs(oldest_measurement_time - newer_key_time)
#                     time_next_to_pose = abs(next_measurement_time - newer_key_time)

#                     if(next_measurement_time < newer_key_time):
#                     # Take care of where next measurement is not past next node
#                         print("popped 2")
#                         self.ts_queue.pop(0)
                    
#                     elif(time_old_to_pose > time_next_to_pose):
#                     # Take care of case where next measurement is better
#                         print("popped 3")
#                         self.ts_queue.pop(0)
#                         print("next meas better")

#                     else:
#                         # Actually add the factor
#                         print("popped 4")
#                         poseKey_to_add = new_key
#                         ts_to_match = self.ts_queue.pop(0)                    
#                         self.last_pose_key = new_key
#                         print("found match")
#             else:
#                 print("in future")
#                 in_future = True
#                 poseKey_to_add = new_key + 1
#                 return poseKey_to_add, None

#         return poseKey_to_add, ts_to_match
    
#     def get_factor_info(self, new_key, poseKey_to_time):
#         ''''
#         new_key = int, The agent posekey id that you will start searching at
#         poseKey_to_time = dict being tracked by factor graph. keys: poseKey values: corresponding timestamp
#         '''

#         # A flag to determine if all remaining measurements are in the future relative to the current pose key's time
#         in_future = False

#         curr_time = poseKey_to_time[new_key] 
#         poseKey_to_add = None
#         msg_to_add = None

#         # If measurement in queue and the oldest measurment is later than current posekey
#         # Process the oldest measurements in self.msg_queue to decide if they are relevant for the current new_key
#         while(len(self.msg_queue) > 1 and in_future is False):       

#             oldest_measurement_time = (self.msg_queue[0].header.stamp.secs * 1_000_000_000 + self.msg_queue[0].header.stamp.nsecs) 
#             next_measurement_time = (self.msg_queue[1].header.stamp.secs * 1_000_000_000 + self.msg_queue[1].header.stamp.nsecs) 
#             # print("Curr", curr_time, "Oldest", oldest_measurement_time, "Next", next_measurement_time)

#             if(oldest_measurement_time < curr_time):
#                 # print('oldest measurement time < curr time')
                
#                 newer_key_time = poseKey_to_time[int(new_key)] 
#                 # print("newer key time: %d\n"%newer_key_time)
#                 older_key_time = poseKey_to_time[int(new_key - 1)]
#                 # print("older key time: %d\n"% older_key_time)
#                 time_to_current = abs(newer_key_time - oldest_measurement_time) 
#                 # print("time to current: %d\n"%time_to_current)
#                 time_to_previous = abs(older_key_time - oldest_measurement_time) 
#                 # print("time to previous: %d\n"%time_to_previous)

#                 if(time_to_current > time_to_previous):
#                     # Meas in queue better suited to previous key, keep working on prev key
#                     # print('time to current is longer than time to prev')
#                     new_key -= 1
#                     if(new_key == self.last_pose_key):
#                         # print('pop')
#                         self.msg_queue.pop(0)
#                 else:
#                     # print('time to current is shortest')

#                     time_old_to_pose = abs(oldest_measurement_time - newer_key_time)
#                     time_next_to_pose = abs(next_measurement_time - newer_key_time)

#                     if(next_measurement_time < newer_key_time):
#                     # Take care of where next measurement is not past next node
#                         self.msg_queue.pop(0)
                    
#                     elif(time_old_to_pose > time_next_to_pose):
#                     # Take care of case where next measurement is better
#                         self.msg_queue.pop(0)
#                         print("next meas better")

#                     else:
#                         # Actually add the factor
#                         poseKey_to_add = new_key
#                         msg_to_add = self.msg_queue.pop(0)                    
#                         self.last_pose_key = new_key
#                         print("setting last pose")
#             else:
#                 in_future = True

#         return poseKey_to_add, msg_to_add


import numpy as np

class TimeSync:
    def __init__(self):
        self.ts_queue = []
        self.last_pose_key = 0

    def add_to_ts_queue(self, t):
        """Add new timestamp to the queue (keeps it sorted)."""
        self.ts_queue.append(t)
        self.ts_queue.sort()

    def get_ts_match(self, new_key, timestamps):
        """
        Find the closest timestamp (past or future) in the queue to the given backbone timestamp.

        Args:
            new_key (int): Index of the current backbone timestamp.
            timestamps (list[float]): List of backbone timestamps.

        Returns:
            (poseKey_to_add, ts_to_match)
                poseKey_to_add: index of backbone timestamp that matched
                ts_to_match: matched timestamp from this topic
        """
        if len(timestamps) == 0 or len(self.ts_queue) == 0:
            return None, None

        curr_time = timestamps[new_key]

        # Compute absolute time differences
        diffs = np.abs(np.array(self.ts_queue) - curr_time)
        min_idx = int(np.argmin(diffs))
        ts_closest = self.ts_queue[min_idx]

        # Found the closest match
        poseKey_to_add = new_key
        ts_to_match = ts_closest

        # Remove the matched timestamp from queue (optional)
        MAX_DT = 0.1  # seconds (or nanoseconds if you're using those)
        if abs(ts_closest - curr_time) > MAX_DT:
            return None, None

        self.ts_queue.pop(min_idx)

        self.last_pose_key = new_key
        return poseKey_to_add, ts_to_match
