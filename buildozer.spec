[app]
title = Android Two Cam
package.name = TwoCam
package.domain = TwoCam.lftr.biz
source.dir = .
source.include_exts = py,png,jpg,kv,atlas
version = 0.1
requirements = python3,kivy,opencv==4.7.0,numpy
orientation = portrait
fullscreen = 0
android.presplash_color = aqua
android.permissions=INTERNET,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE,MANAGE_EXTERNAL_STORAGE,CAMERA
android.archs = arm64-v8a
p4a.requirements = python3,kivy,opencv==4.7.0,numpy
p4a.setup_py = true
[buildozer]
log_level = 2
warn_on_root = 1