"""Own one Windows process tree, without process-name enumeration or taskkill.

Create suspended, assign to a non-breakaway job, then resume: even the venv
redirector cannot spawn an unowned descendant in the assignment race.
Layout: Microsoft JOBOBJECT_BASIC_LIMIT_INFORMATION / EXTENDED_LIMIT_INFORMATION.
"""
import ctypes as C
from ctypes import wintypes as W
import subprocess
import _winapi


class Basic(C.Structure):
    _fields_=[('process_time',C.c_int64),('job_time',C.c_int64),('flags',W.DWORD),
              ('min_working',C.c_size_t),('max_working',C.c_size_t),('active',W.DWORD),
              ('affinity',C.c_size_t),('priority',W.DWORD),('scheduling',W.DWORD)]


class Extended(C.Structure):
    _fields_=[('basic',Basic),('io',C.c_uint64*6),('process_memory',C.c_size_t),
              ('job_memory',C.c_size_t),('peak_process',C.c_size_t),('peak_job',C.c_size_t)]


K=C.WinDLL('kernel32',use_last_error=True)
for name,args,result in [
    ('CreateJobObjectW',[C.c_void_p,W.LPCWSTR],W.HANDLE),
    ('SetInformationJobObject',[W.HANDLE,C.c_int,C.c_void_p,W.DWORD],W.BOOL),
    ('AssignProcessToJobObject',[W.HANDLE,W.HANDLE],W.BOOL),
    ('TerminateJobObject',[W.HANDLE,W.UINT],W.BOOL),
    ('ResumeThread',[W.HANDLE],W.DWORD),('CloseHandle',[W.HANDLE],W.BOOL)]:
    fn=getattr(K,name);fn.argtypes=args;fn.restype=result


def checked(value):
    if not value:raise C.WinError(C.get_last_error())
    return value


class OwnedTree:
    def __init__(self,argv):
        self.argv=argv;self.process=None;self.job=checked(K.CreateJobObjectW(None,None))
        thread=None
        try:
            limits=Extended();limits.basic.flags=0x2000
            checked(K.SetInformationJobObject(self.job,9,C.byref(limits),C.sizeof(limits)))
            si=subprocess.STARTUPINFO()
            si.dwFlags|=subprocess.STARTF_USESTDHANDLES
            si.hStdInput=_winapi.GetStdHandle(_winapi.STD_INPUT_HANDLE)
            si.hStdOutput=_winapi.GetStdHandle(_winapi.STD_OUTPUT_HANDLE)
            si.hStdError=_winapi.GetStdHandle(_winapi.STD_ERROR_HANDLE)
            self.process,thread,self.pid,_=_winapi.CreateProcess(
                None,subprocess.list2cmdline(argv),None,None,True,
                0x00000004 | 0x08000000,None,None,si)
            checked(K.AssignProcessToJobObject(self.job,self.process))
            if K.ResumeThread(thread)==0xffffffff:raise C.WinError(C.get_last_error())
        except BaseException:
            if self.process is not None:
                _winapi.TerminateProcess(self.process,2)
                _winapi.WaitForSingleObject(self.process,5000)
            self.close()
            raise
        finally:
            if thread is not None:_winapi.CloseHandle(thread)

    def wait(self,timeout=None):
        milliseconds=_winapi.INFINITE if timeout is None else max(0,int(timeout*1000))
        result=_winapi.WaitForSingleObject(self.process,milliseconds)
        if result==_winapi.WAIT_TIMEOUT:raise subprocess.TimeoutExpired(self.argv,timeout)
        if result!=_winapi.WAIT_OBJECT_0:raise RuntimeError(f'Unexpected wait result: {result}')
        return _winapi.GetExitCodeProcess(self.process)

    def kill(self):checked(K.TerminateJobObject(self.job,2))

    def close(self):
        if self.job is not None:K.CloseHandle(self.job);self.job=None
        if self.process is not None:_winapi.CloseHandle(self.process);self.process=None
