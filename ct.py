import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import interp1d

def solve_control_problem(T, dt, a1, b1, a2, b2, sigma, mu, n1_T, n2_T):
    
    def RK4(f):
        return lambda s, v, ds: (
                lambda dy1: (
                lambda dy2: (
                lambda dy3: (
                lambda dy4: (dy1 + 2*dy2 + 2*dy3 + dy4)/6
                )( ds * f( s + ds  , v + dy3   ) )
                )( ds * f( s + ds/2, v + dy2/2 ) )
                )( ds * f( s + ds/2, v + dy1/2 ) )
                )( ds * f( s       , v         ) )
    
    
    e = [
        lambda s, v: 0.25 * 1/((sigma**2) * (mu**2)) * (
            8*a1(s)*v[0]*(mu**2)*(sigma**2) + 4*b1(s)*v[1]*(mu**2)*(sigma**2) - 4*(v[0]**2)*(mu**2) - 
            (v[1]**2)*(sigma**2) + 4*(sigma**2)*(mu**2)
        ),
        lambda s, v: 1/((sigma**2) * (mu**2)) * (
            a1(s)*v[1]*(mu**2)*(sigma**2) + 2*b1(s)*v[2]*(sigma**2)*(mu**2) + 2*v[0]*b2(s)*(mu**2)*(sigma**2) + 
            v[1]*a2(s)*(mu**2)*(sigma**2) - v[1]*v[0]*(mu**2) - v[2]*v[1]*(sigma**2)
        ),
        lambda s, v: -0.25 * 1./((sigma**2) * (mu**2)) * (
            -4*v[1]*b2(s)*(mu**2)*(sigma**2) - 8*v[2]*a2(s)*(mu**2)*(sigma**2) + (v[1]**2)*(mu**2) + 
            4*(v[2]**2)*(sigma**2) - 4*(sigma**2)*(mu**2)
        ),
        lambda s, v: 0.5 * 1/((sigma**2) * (mu**2)) * (
            2*v[3]*a1(s)*(mu**2)*(sigma**2) + 2*v[4]*b1(s)*(mu**2)*(sigma**2) - 2*v[3]*v[0]*(mu**2) - 
            v[4]*v[1]*(sigma**2)
        ),
        lambda s, v: -0.5 * 1/((sigma**2) * (mu**2)) * (
            -2*v[3]*(mu**2)*(sigma**2)*b2(s) - 2*v[4]*a2(s)*(mu**2)*(sigma**2) + v[3]*v[1]*(mu**2) + 
            2*v[4]*v[2]*(sigma**2)
        ),
        lambda s, v: -0.25 * 1/((sigma**2) * (mu**2)) * (
            (v[3]**2)*(mu**2) + (v[4]**2)*(sigma**2)
        ),        
    ]
    
    e0 = [
        lambda s, v: -(1./36.)*(v[1]**2) + (5./2.)*v[1]*math.sin(s) + 5*v[1]*math.cos(s) - (1./16.)*(v[0]**2) + 1,
        lambda s, v: -(1./9.)*v[1]*v[2] + (5./2.)*v[1]*math.cos(s) - (1./16.)*v[0]*v[1] + 5*v[2]*math.sin(s) + 
        (3./2.)*v[1] + 5*v[0],
        lambda s, v: -(1./64.)*v[1]**2 - (1./9.)*v[2]**2 + (5./2.)*v[1] + 3*v[2] + 1,
        lambda s, v: -(1./18.)*v[1]*v[4] + (5./2.)*v[4]*math.sin(s) + (5./2.)*v[3]*math.cos(s) - (1./16.)*v[0]*v[3],
        lambda s, v: -(1./9.)*v[2]*v[4] - (1./32.)*v[1]*v[3] + (3./2.)*v[4] + (5./2.)*v[3],
        lambda s, v: -(1./36.)*v[4]**2 - (1./64.)*v[3]**2        
    ]
       
    def f(s, v):        
        return np.array([e[0](s, v), e[1](s, v), e[2](s, v), e[3](s, v), e[4](s, v), e[5](s, v)], dtype=np.float_)
        #return np.array([e0[0](s, v), e0[1](s, v), e0[2](s, v), e0[3](s, v), e0[4](s, v), e0[5](s, v)], dtype=np.float_)        
    
    dv = RK4(lambda s, v: f(s, v))
    
    t = np.arange(0, T+dt, dt)
    v0 = np.array([1., 0., 1., 0., 0., 0.], dtype=np.float_)
    v = [v0]
    sprev = 0
    vprev = v0
    for s in t[1:]:
        ds = abs(s-sprev)
        vnew = vprev + dv(s, vprev, ds)
        
#        print(s, vnew)
        sprev = s
        v.append(vnew)
        vprev = vnew
    
    #plt.subplot(2, 1, 1)
    plt.plot(t, v)
    plt.show()
    
    n1 = []
    n2 = []
    curn1 = n1_T
    curn2 = n2_T        
    sprev = T
    
    for i in reversed(range(len(t))):
        ds = abs(t[i] - sprev)
                
        vn1 = -a1(s)*curn1 - b1(s)*curn2 + 0.5 * (2*v[i][0]*curn1 + v[i][1]*curn2 + v[i][3]) / sigma**2
        vn2 = -b1(s)*curn1 - a2(s)*curn2 + 0.5 * (v[i][1]*curn1 + 2*v[i][2]*curn2 + v[i][4]) / mu**2
        
        curn1 = curn1 - ds * vn1
        curn2 = curn2 - ds * vn2
        
        n1.append(curn1)
        n2.append(curn2)
    
    n1, n2 = n1[::-1], n2[::-1]
    #plt.subplot(2, 1,2)
    plt.plot(t, n1)
    plt.plot(t, n2)
    plt.show()
    
def bellman_dynamic(
    f0, phi, f, 
    vt, vn1, vn2, 
    n1_T, n2_T):
    
    def integral(f0, a, b, npoints):
        h = np.abs(b - a)/npoints
        time = np.arange(a, b, h)
        S = 0.0
        for t in time[:time.shape[0]-1]:
            S += h/2 * (f0(t) + f0(t+h))
    
    def find_control(F, n0, n1, t0, t1):
        # dn/dt = F(t, n) + u(t)
        # n1 = n0 + (t1 - t0) * (F(t, n) + u(t))
        # u(t) = (n1 - n0) / (t1 - t0) - F(t, n)        
        return (n1 - n0) / (t1 - t0) - F(t, n0)
    
    def find_optimal_control(t0, t1, n1, n2, n1_to, n2_to, f, f0, next_bellman):        
        
        best_local_control = None
        best_bellman_value = -100000
               
        for m1 in range(n1_to.shape[0]):
            for m2 in range(n2_to.shape[0]):
                z, z_next = np.array([n1, n2]), np.array([n1_to[m1], n2_to[m2]])
                        
                u_local = find_control(f, z, z_next, t0, t1)
                quality = integral(lambda t: f0(u, x, t), z, z_next, 5) + next_bellman[m1][m2]
                        
                if quality > best_bellman_value: 
                    best_bellman_value = quality
                    best_local_control = u_local
                
        return best_local_control, best_bellman_value
                
    
    bellman_values = np.ndarray((vt.shape[0]-1, vn1.shape[0], vn2.shape[0]), dtype=np.float_)
    optimal_control = np.ndarray((vt.shape[0]-1, vn1.shape[0], vn2.shape[0]), dtype=np)
    
    # backward:
    
    # calculate bellman for T_{n-1} 
    last_bellman = [[0]]
    for i in range(vn1.shape[0]):
        for j in range(vn2.shape[0]):
            n1, n2 = vn1[i], vn2[j]
            t0, t1 = vt[-2], vt[-1]
            u, b = find_optimal_control(t0, t1, n1, n2, np.array([n1_T]), np.array([n2_T]), f, f0, last_bellman)
            bellman_values[-1][i][j] = b
        
    # from T_{n-2} to T_{0}
    for i in reversed(range(vt.shape[0]-1)):
        for i in range(vn1.shape[0]):
            for j in range(vn2.shape[0]):
                n1, n2 = vn1[i], vn2[j]
                t0, t1 = vt[i], vt[i+1]
                u, b = find_optimal_control(t0, t1, n1, n2, np.array([n1_T]), np.array([n2_T]), f, f0, last_bellman)
                bellman_values[-1][i][j] = b
    
    # forward:
    best_n10 = None
    best_n20 = None
    best_bellman = -100000
    
    # find first point:
    for i in range(vn1.shape[0]):
        for j in range(vn2.shape[0]):
            n1, n2 = vn1[i], vn2[j]
            if bellman_values[0][i][j] + phi(np.array([n1, n2])) > best_bellman:
                best_n10, best_n20 = n1, n2
                        
    # find other points:
    best_n1, best_n2 = [best_n10], [best_n20]
    for i in vt[1:]:
        z = np.array[best_n1[-1], best_n2[-1]]
#        next_z = z + (vt[i] - vt[i-1])*(f(z) + )
        best_n1.append(next_z[0])
        best_n2.append(next_z[1])
        
    plt.plot(best_n1)
    plt.plot(best_n2)
    plt.show()

def bellman_dynamic_general(f0, f, phi, vt, vn, n_T):
    print('begin')
    
    def calc_integral(f0, a, b, npoints):
        h = np.abs(b - a)/npoints
        time = np.arange(a, b, h)
        S = 0.0
        for t in time[:time.shape[0]-1]:
            S += h/2 * (f0(t) + f0(t+h))
            
    def find_control(F, n0, n1, t0, t1):
        # dn/dt = F(t, n) + u(t)
        # n1 = n0 + (t1 - t0) * (F(t, n) + u(t))
        # u(t) = (n1 - n0) / (t1 - t0) - F(t, n)        
        return (n1 - n0) / (t1 - t0) - F(t0, n0)
    
    def find_optimal_control(t0, t1, n, vn, f, f0, next_bellman):        
        assert t0 < t1        
        assert next_bellman.shape[0] == vn.shape[0]
        
        best_local_control = None
        optimal_next_index = None
        optimal_next_step = None
        best_bellman_value = 1000000
                
        for i in range(vn.shape[0]):
            z, z_next = n, vn[i]
            u_local = find_control(f, z, z_next, t0, t1)
            int_value = 1.0*(t1 - t0) * (f0(u_local, z, t0) + f0(u_local, z_next, t1))/2.0
            quality = int_value + next_bellman[i]
            
            #print(quality, best_bellman_value)
            if quality <= best_bellman_value: 
                optimal_next_index = i
                best_bellman_value = quality
                best_local_control = u_local
                optimal_next_step = z_next
                
        return best_local_control, best_bellman_value, optimal_next_step, optimal_next_index
    
    bellman_values = np.ndarray((vt.shape[0]-1, vn.shape[0]), dtype=np.float_)
    optimal_control = np.array([vn.tolist()]*(vt.shape[0]-1))
    optimal_next_index = np.ndarray((vt.shape[0]-1, vn.shape[0]), dtype=int)
    
    # backward:
    
    print('calculate bellman for T_{n-1}')
    # calculate bellman for T_{n-1} 
    last_bellman = np.array([0])
    for i in range(vn.shape[0]):
        n = vn[i]
        t0, t1 = vt[-2], vt[-1]        
        u, b, n_next, i_next = find_optimal_control(t0, t1, n, np.array([n_T]), f, f0, last_bellman)
        bellman_values[-1][i] = b
        optimal_control[-1][i] = u        
        optimal_next_index[-1][i] = i_next

    print('from T_{n-2} to T_{0}')
    #print(optimal_next_index[19])
    # from T_{n-2} to T_{0}
    
          
    for t in reversed(range(vt.shape[0]-1)):        
        for i in range(vn.shape[0]):            
            n = vn[i]
            t0, t1 = vt[t], vt[t+1]
            u, b, n_next, i_next = find_optimal_control(t0, t1, n, vn, f, f0, bellman_values[t])
            
            bellman_values[t][i] = b
            optimal_control[t][i] = u
            optimal_next_index[t][i] = i_next
    
    # forward:    
    best_i0 = None
    best_n0 = None    
    best_bellman = 1000000
            
    print('find first point:')
    # find first point:
    for i in range(vn.shape[0]):
        n = vn[i]
        if bellman_values[0][i] + phi(n) <= best_bellman:
            best_i0, best_n0 = i, n
            best_bellman = bellman_values[0][i] + phi(n)
    
    print('find other points:')
    # find other points:
    best_i, best_n = [best_i0], [best_n0]
    for t in vt[:vt.shape[0]-2]:
        i, n = best_i[-1], best_n[-1]
        next_i = optimal_next_index[t][i]        
        next_n = vn[i]        
        #next_n = n + (vt[t] - vt[t-1])*(f(vt[t-1], n) + u)
        best_n.append(next_n), best_i.append(next_i)
    best_n.append(n_T)
    
    n1, n2 = [], []
    for n in best_n:
        #print(n)
        n1.append(n[0]), n2.append(n[1])        
    
    n1, n2 = np.array(n1), np.array(n2)
    #print(n1.shape, n2.shape, vt.shape)
    
    nt = np.linspace(0, 1, num=100, endpoint=True)
    f1 = interp1d(vt, n1, kind='cubic')
    f2 = interp1d(vt, n2, kind='cubic')    
    
    #plt.plot(vt, n1)
    plt.plot(nt, f1(nt))    
    #plt.plot(vt, n2)
    plt.plot(nt, f2(nt))
    plt.show()
def demo_general():
    def a1(s):
        return 3*math.cos(1.5*s)
        
    def b1(s):
        return 2*math.sin(s)
    
    def a2(s):
        return 1
    
    def b2(s):
        return 1
    
    def f0(g, n, t):
        assert g.shape == n.shape == (2, )
        return (n[0])**2 + (n[1])**2 +(g[0])**2 + (g[1])**2
    
    def f(t, n):
        assert n.shape == (2, )
        return np.array([a1(t)*n[0] + b1(t)*n[1], b2(t)*n[0] + a2(t)*n[1], ])
    
    def phi(n):
        assert n.shape == (2, )
        return n[0]**2 + n[1]**2
    
    vn1, vn2 = np.arange(0, 100, 10), np.arange(0, 100, 10)
    vn = []
    for n1 in vn1:
        for n2 in vn2:
            vn.append(np.array([n1, n2], dtype=np.float_)) 
    
    None
    vn = np.array(vn)
    
    vt = np.arange(0, 1+0.05, 0.05)
    n_T = np.array([60, 60])
    
    
    bellman_dynamic_general(f0, f, phi, vt, vn, n_T )   
