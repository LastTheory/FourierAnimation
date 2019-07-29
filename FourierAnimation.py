###################################################
# Creates an animation of a complex Fourier series reconstruction of a contour.
# See demo: https://youtu.be/rcA26UL32Pc
# Example code usage at bottom
#
# Copyright (c) 2019 Last Theory Productions, LLC
###################################################

import numpy as np
import cv2
from scipy.interpolate import CubicSpline

# parametrizes list of (x, y) points in terms of distance along the curve
# pts: list of  ordered points (n x 2) that define your curve
# periodic: is the computation over a closed curve?
# returns: cumulative distance along curve between points, list of points curve goes through (which may have been modified)
def distance(pts, periodic=False):
    if periodic:
        pts = np.vstack((pts, pts[0]))
    d = np.zeros((len(pts),))
    for i in range(len(d)):
        if i == 0:
            d[i] = 0
        else:
            d[i] = d[i - 1] + np.linalg.norm(pts[i] - pts[i - 1])
    d /= d[-1]
    return d, pts

#interpolates points based on function
# x: independent variable
# y: dependent variable
# returns: spline function
def spline_func(x, y, periodic=False):
    if periodic:
        spline = CubicSpline(x, y, bc_type='periodic')
    else:
        spline = CubicSpline(x, y)
    return spline

# for a function f(t) of points, computes the complex Fourier coefficients
# pts: numpy array of ordered points (n x 2) that define your curve
# nvec: number of Fourier components to calculate. Accuracy and computational complexity increases with this
# nt: number of sample points for spline. The higher this is, the more accurate the coefficients will be
# returns: Fourier coefficients, corresponding n for each coefficient (from -nvec to nvec)
def calculateFourierSeries(pts, nvec=10, nt=1e5):
    d, pts = distance(pts, periodic=True) #parametrize
    x_spline = spline_func(d, pts[:, 0], periodic=True) #create function x(t)
    y_spline = spline_func(d, pts[:, 1], periodic=True) #create function y(t)

    t = np.linspace(0, 1, num=(nt+1))
    nlist = np.linspace(-nvec, nvec, num=2*nvec+1).astype(np.int32)
    c = np.zeros((len(nlist),), dtype=np.complex)
    for n in nlist:
        exp = np.exp(-1j * n * 2 * np.pi * t) * (1/nt)
        f = (x_spline(t) + y_spline(t) * 1j) * exp
        c[n+nvec] = np.sum(f)
    return c, nlist

# given Fourier coefficients, computes each compenent at a given time
# coef: Fourier coefficients
# n: n corresponding to each Fourier coefficient (-nvec to nvec)
# t: time step at which to calculate Fourier series
# returns: cumulative sum of each component of the series (location of arrow tips) at a time t
def calcSums(coef, n, t):
    nvec = np.abs(n[0])
    fvec = np.zeros((len(n),), dtype=complex)
    for i in range(nvec + 1):
        if i == 0:
            fvec[i] = coef[nvec]
        else:
            fvec[2*i - 1] = coef[nvec + i] * np.exp(1j * n[nvec + i] * 2 * np.pi * t)
            fvec[2*i] = coef[nvec - i] * np.exp(1j * n[nvec - i] * 2 * np.pi * t)
    return fvec.cumsum()

# create frame animations
# c: Fourier coefficients
# nlist: list of n for each Fourier coefficient (-nvec to nvec)
# imshape: tuple size of output image (h, w). Contour points (with some margin) must fit inside this size
# animtime: number of timesteps to draw video. animtime / FPS of video is approximate length of end video
# ptdensity: number of points to draw along trajectory curve. Higher density, smoother the line
# display: whether to display frames as they are processed
# save: whether to save video. If true, a valid vidwriter must be passed
# vidwriter: video writer to save video
# drawArrows: whether to draw arrows representing components
# drawCircles: whether to draw circles around arrow components
# returns last image of fully drawn contour
def animate(c, nlist, imshape, animtime=1e3, ptdensity=200, display=True, save=False, vidwriter=None, drawArrows=True, drawCircles=True):
    time = np.linspace(0, 1, num=animtime + 1)
    endpoints = []
    arrowColor = (200, 180, 195)
    trajColor = (255, 161, 105)
    circleColor = (115, 115, 115)
    for t in time:
        print(t)
        img = np.zeros((imshape[0], imshape[1], 3))
        sums = calcSums(c, nlist, t)
        for i, sum in enumerate(sums):
            if i == 0:
                continue
            pt1 = (int(np.real(sums[i-1])), int(np.imag(sums[i-1])))
            pt2 = (int(np.real(sum)), int(np.imag(sum)))
            if drawArrows:
                cv2.arrowedLine(img, pt1, pt2, arrowColor, 2, tipLength=0.05)
            if drawCircles:
                radius = int(round(np.sqrt((pt2[1] - pt1[1])*(pt2[1] - pt1[1]) + (pt2[0] - pt1[0])*(pt2[0] - pt1[0]))))
                cv2.circle(img, pt1, radius, circleColor, thickness=1)

        endpoints.append(np.array([np.real(sums[-1]), np.imag(sums[-1])]))
        r = np.linspace(0, 1, ptdensity*len(endpoints))
        if t == 0:
            endp = np.round(endpoints[0]).astype(np.int32)
            cv2.circle(img, (endp[0], endp[1]), 1, trajColor, thickness=-1)
        else:
            dist, endp = distance(np.asarray(endpoints))
            ex = spline_func(dist, endp[:, 0])
            ey = spline_func(dist, endp[:, 1])
            trajectory = np.round(np.vstack((ex(r), ey(r))).T).astype(np.int32)
            for traj in trajectory:
                cv2.circle(img, (traj[0], traj[1]), 1, trajColor, thickness=-1)
        if save:
            vidwriter.write(np.uint8(img))
        if display:
            cv2.imshow('Fourier', cv2.resize(np.uint8(img), (imshape[1]//3, imshape[0]//3)))
            cv2.waitKey(1)
    if save:
        vidwriter.release()
    return img

###########################################
# Example: (contour is n x 2 list of (x, y) points for your contour)
#
# import FourierAnimation as fa
# import cv2
#
# contour = <your n x 2 array of ordered (x, y) contour points>
# imshape = <(h x w) size of output image>
# c, nlist = fa.calculateFourierSeries(contour, nvec=100) #calculate Fourier coefficients
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# out = cv2.VideoWriter('fourier.avi', fourcc, 30, (imshape[1], imshape[0])) #video writer
# fa.animate(c, nlist, imshape, save=True, vidwriter=out)
