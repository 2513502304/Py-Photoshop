'''
Author: 未来可欺 2513502304@qq.com
Date: 2023-12-17 23:22:19
LastEditors: 未来可欺 2513502304@qq.com
LastEditTime: 2025-07-31 01:53:07
Python version: 3.11.3 64-bit
Description: Main file for Photoshop application
'''

import sys
import photoshop as ps

if __name__ == "__main__":
    app = ps.QApplication([])

    mainwindow = ps.Photoshop()
    mainwindow.resize(1200, 800)
    mainwindow.show()

    sys.exit(app.exec())
