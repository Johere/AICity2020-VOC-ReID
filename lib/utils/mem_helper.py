# -*- coding: utf-8 -*-
import psutil


class CPUMemory:
    def __init__(self, unit='G'):
        self.memory = psutil.virtual_memory()
        self.unit = unit
        if unit == 'G':
            self.factor = 1024 * 1024 * 1024
        elif unit == 'M':
            self.factor = 1024 * 1024
        elif unit == 'M':
            self.factor = 1
        else:
            raise ValueError('Unknown memory unit:{}'.format(unit))

    def flush(self):
        self.memory = psutil.virtual_memory()

    @property
    def total_mem(self):
        # 系统总计内存
        return float(self.memory.total) / self.factor, 4

    @property
    def used_mem(self):
        # 系统已经使用内存
        return float(self.memory.used) / self.factor

    @property
    def free_mem(self):
        # 系统空闲内存
        return float(self.memory.free) / self.factor
