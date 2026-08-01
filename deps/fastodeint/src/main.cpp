#include <pybind11/pybind11.h>
#include <cmath>
#include <boost/array.hpp>

#include <boost/numeric/odeint.hpp>
#include <numeric>
#include <pybind11/numpy.h>

using namespace std;
using namespace boost::numeric::odeint;
namespace py = pybind11;

typedef boost::array< double , 33 > state_type;
boost::array<double, 31> params{};

void pensim_func(const state_type &x , state_type &dxdt , const double t ){
    double mu_p = params[0];
    double mux_max = params[1];

    double ratio_mu_e_mu_b = 0.4;
    double P_std_dev = 0.0015;
    double mean_P = 0.002;
    double mu_v = 1.71e-4;
    double mu_a = 3.5e-3;
    double mu_diff = 5.36e-3;
    double beta_1 = 0.006;
    double K_b = 0.05;
    double K_diff = 0.75;
    double K_diff_L = 0.09;
    double K_e = 0.009;
    double K_v = 0.05;
    double delta_r = 0.75e-4;
    double k_v = 3.22e-5;
    double D = 2.66e-11;
    double rho_a0 = 0.35;
    double rho_d = 0.18;
    double mu_h = 0.003;
    double r_0 = 1.5e-4;
    double delta_0 = 1e-4;

    // Process related parameters
    double Y_sX = 1.85;
    double Y_sP = 0.9;
    double m_s = 0.029;
    double c_oil = 1000;
    double c_s = 600;
    double Y_O2_X = 650;
    double Y_O2_P = 160;
    double m_O2_X = 17.5;
    double alpha_kla = params[2];

    double a = 0.38;
    double b = 0.34;
    double c = -0.38;
    double d = 0.25;
    double Henrys_c = 0.0251;
    double n_imp = 3;
    double r = 2.1;
    double r_imp = 0.85;
    double Po = 5;
    double epsilon = 0.1;
    double g = 9.81;
    double R = 8.314;
    double X_crit_DO2 = 0.1;
    double P_crit_DO2 = 0.3;
    double A_inhib = 1;
    double Tf = 288;
    double Tw = 288;
    double Tcin = 285;
    double Th = 333;
    double Tair = 290;
    double C_ps = 5.9;
    double C_pw = 4.18;
    double dealta_H_evap = 2430.7;
    double U_jacket = 36;
    double A_c = 105;
    double Eg = 14880;
    double Ed = 173250;
    double k_g = 450;
    double k_d = 2.5e+29;
    double Y_QX = 25;
    double abc = 0.033;
    double gamma1 = 0.0325e-5;
    double gamma2 = 2.5e-11;
    double m_ph = 0.0025;
    double K1 = 1e-5;
    double K2 = 2.5e-8;
    double N_conc_oil = 20000;
    double N_conc_paa = params[3];

    double N_conc_shot = 400000;
    double Y_NX = 10;
    double Y_NP = 80;
    double m_N = 0.03;
    double X_crit_N = 150;
    double PAA_c = params[4];

    double Y_PAA_P = 187.5;
    double Y_PAA_X = 45;
    double m_PAA = 1.05;
    double X_crit_PAA = 2400;
    double P_crit_PAA = 200;
    double B_1 = -64.29;
    double B_2 = -1.825;
    double B_3 = 0.3649;
    double B_4 = 0.1280;
    double B_5 = -4.9496e-04;
    double delta_c_0 = 0.89;
    double k3 = 0.005;
    double k1 = 0.001;
    double k2 = 0.0001;
    double t1 = 1;
    double t2 = 250;
    double q_co2 = 0.1353;
    double X_crit_CO2 = 7570;
    double alpha_evp = 5.2400e-4;
    double beta_T = 2.88;
    double pho_g = 1540;
    double pho_oil = 900;
    double pho_w = 1000;
    double pho_paa = 1000;
    double O_2_in = 0.21;
    double N2_in = 0.79;
    double C_CO2_in = 0.033;
    double Tv = 373;
    double T0 = 273;
    double alpha_1 = 2451.8;

    // process inputs
    double inhib_flag = params[5];

    double Fs = params[6];

    double Fg = (params[7] / 60);

    double RPM = params[8];

    double Fc = params[9];

    double Fh = params[10];

    double Fb = params[11];

    double Fa = params[12];

    double step1 = params[13];

    double Fw = params[14];

    if (Fw < 0) {
        Fw = 0;
    }
    double pressure = params[15];

    // Viscosity flag
    double viscosity = params[16];

    if (params[30] == 0) {
        viscosity = x[9];
    }

    double F_discharge = params[17];
    double Fpaa = params[18];
    double Foil = params[19];
    double NH3_shots = params[20];
    double dist_flag = params[21];
    double distMuP = params[22];
    double distMuX = params[23];
    double distsc = params[24];
    double distcoil = params[25];
    double distabc = params[26];
    double distPAA = params[27];
    double distTcin = params[28];
    double distO_2_in = params[29];

    double pho_b = (1100 + x[3] + x[11] + x[12] + x[13] + x[14]);

    if (dist_flag == 1) {
        mu_p += distMuP;
        mux_max += distMuX;
        c_s = c_s + distsc;
        c_oil += distcoil;
        abc += distabc;
        PAA_c += distPAA;
        Tcin += distTcin;
        O_2_in += distO_2_in;
    }

    // Process parameters
    // Adding in age-dependant term
    double A_t1 = (x[10]) / (x[11] + x[12] + x[13] + x[14]);

    // Variables
    double s = x[0];
    double a_1 = x[12];
    double a_0 = x[11];
    double a_3 = x[13];
    double total_X = x[11] + x[12] + x[13] + x[14];  // Total Biomass

    // Calculating liquid height in vessel
    double h_b = (x[4] / 1000) / (3.141592653589793 * pow(r, 2));
    h_b = h_b * (1 - epsilon);

    // Calculating log mean pressure of vessel
    double pressure_bottom = 1 + pressure + pho_b * h_b * 9.81e-5;
    double pressure_top = 1 + pressure;
    double total_pressure = (pressure_bottom - pressure_top) / (log(pressure_bottom / pressure_top));

    // Ensuring minimum value for viscosity
    if (viscosity < 4) {
        viscosity = 1;
    }
    double DOstar_tp = total_pressure * O_2_in / Henrys_c;
    double pH_inhib;
    double NH3_inhib;
    double T_inhib;
    double DO_2_inhib_X;
    double DO_2_inhib_P;
    double CO2_inhib;
    double PAA_inhib_X;
    double PAA_inhib_P;

    // Inhibition flags
    if (inhib_flag == 0) {
        pH_inhib = 1;
        NH3_inhib = 1;
        T_inhib = 1;
        mu_h = 0.003;
        DO_2_inhib_X = 1;
        DO_2_inhib_P = 1;
        CO2_inhib = 1;
        PAA_inhib_X = 1;
        PAA_inhib_P = 1;
    }

    double pH;

    if (inhib_flag == 1) {
        pH_inhib = (1 / (1 + (x[6] / K1) + (K2 / x[6])));
        NH3_inhib = 1;
        T_inhib = (k_g * exp(-(Eg / (R * x[7]))) - k_d * exp(-(Ed / (R * x[7])))) * 0 + 1;
        CO2_inhib = 1;
        DO_2_inhib_X = 0.5 * (1 - tanh(A_inhib * (X_crit_DO2 * ((total_pressure * O_2_in) / Henrys_c) - x[1])));
        DO_2_inhib_P = 0.5 * (1 - tanh(A_inhib * (P_crit_DO2 * ((total_pressure * O_2_in) / Henrys_c) - x[1])));
        PAA_inhib_X = 1;
        PAA_inhib_P = 1;
        pH = -log10(x[6]);
        mu_h = exp((B_1 + B_2 * pH + B_3 * x[7] + B_4 * (pow(pH, 2))) + B_5 * (pow(x[7],2)));
    }

    pH_inhib = 1 / (1 + (x[6] / K1) + (K2 / x[6]));
    NH3_inhib = 0.5 * (1 - tanh(A_inhib * (X_crit_N - x[30])));
    T_inhib = k_g * exp(-(Eg / (R * x[7]))) - k_d * exp(-(Ed / (R * x[7])));
    CO2_inhib = 0.5 * (1 + tanh(A_inhib * (X_crit_CO2 - x[28] * 1000)));
    DO_2_inhib_X = 0.5 * (1 - tanh(A_inhib * (X_crit_DO2 * ((total_pressure * O_2_in) / Henrys_c) - x[1])));
    DO_2_inhib_P = 0.5 * (1 - tanh(A_inhib * (P_crit_DO2 * ((total_pressure * O_2_in) / Henrys_c) - x[1])));
    PAA_inhib_X = 0.5 * (1 + (tanh((X_crit_PAA - x[29]))));
    PAA_inhib_P = 0.5 * (1 + (tanh((-P_crit_PAA + x[29]))));
    pH = -log10(x[6]);
    mu_h = exp((B_1 + B_2 * pH + B_3 * x[7] + B_4 * (pow(pH, 2))) + B_5 * (pow(x[7],2)));

    // Main rate equations for kinetic expressions
    // Penicillin inhibition curve
    double P_inhib = 2.5 * P_std_dev * (
            pow((P_std_dev * 2.5066282746310002), -1) * exp(-0.5 * pow(((s - mean_P) / P_std_dev), 2)));

    // Specific growth rates of biomass regions with inhibition effect
    double mu_a0 = ratio_mu_e_mu_b * mux_max * pH_inhib * NH3_inhib * T_inhib * DO_2_inhib_X * CO2_inhib * PAA_inhib_X;

    // Rate constant for Branching A0
    double mu_e = mux_max * pH_inhib * NH3_inhib * T_inhib * DO_2_inhib_X * CO2_inhib * PAA_inhib_X;

    // Rate constant for extension A1
    K_diff = 0.75 - (A_t1 * beta_1);
    if (K_diff < K_diff_L) {
        K_diff = K_diff_L;
    }
    // Growing A_0 region
    double r_b0 = mu_a0 * a_1 * s / (K_b + s);
    double r_sb0 = Y_sX * r_b0;

    // Non-growing regions A_1 region
    double r_e1 = (mu_e * a_0 * s) / (K_e + s);
    double r_se1 = Y_sX * r_e1;

    // Differentiation (A_0 -> A_1)
    double r_d1 = mu_diff * a_0 / (K_diff + s);
    double r_m0 = m_s * a_0 / (K_diff + s);

    int n = 16;
    boost::array<double, 10> phi{};
    phi[0] = x[26];

    for (int k = 2; k < 11; k++) {
        phi[k - 1] = 4.1887902047863905 * pow((1.5e-4 + (k - 2) * delta_r), 3) * x[n] * delta_r;
        n += 1;
    }

    // Total vacuole volume
    double v_2 = accumulate(phi.begin(), phi.end(), 0.0);
    double rho_a1 = (a_1 / ((a_1 / rho_a0) + v_2));
    double v_a1 = a_1 / (2 * rho_a1) - v_2;

    // Penicillin produced from the non-growing regions  A_1 regions
    double r_p = mu_p * rho_a0 * v_a1 * P_inhib * DO_2_inhib_P * PAA_inhib_P - mu_h * x[3];

    // ----- Vacuole formation-------
    double r_m1 = (m_s * rho_a0 * v_a1 * s) / (K_v + s);

    // ------ Vacuole degeneration -------------------
    double r_d4 = mu_a * a_3;

    // ------ Vacuole Volume -------------------
    // n_0 - mean vacoule number density for vacuoles sized ranging from delta_0 -> r_0
    double dn0_dt = ((mu_v * v_a1) / (K_v + s)) * (1.909859317102744 * pow((r_0 + delta_0), -3)) - k_v * x[15];

    n = 16;
    // n_j - mean vacoule number density for vacuoles sized ranging from r_{j}
    double dn1_dt = -k_v * ((x[n + 1] - x[n - 1]) / (2 * delta_r)) + D * (x[n + 1] - 2 * x[n] + x[n - 1]) / pow(delta_r, 2);
    n += 1;
    double dn2_dt = -k_v * ((x[n + 1] - x[n - 1]) / (2 * delta_r)) + D * (x[n + 1] - 2 * x[n] + x[n - 1]) / pow(delta_r, 2);
    n += 1;
    double dn3_dt = -k_v * ((x[n + 1] - x[n - 1]) / (2 * delta_r)) + D * (x[n + 1] - 2 * x[n] + x[n - 1]) / pow(delta_r, 2);
    n += 1;
    double dn4_dt = -k_v * ((x[n + 1] - x[n - 1]) / (2 * delta_r)) + D * (x[n + 1] - 2 * x[n] + x[n - 1]) / pow(delta_r, 2);
    n += 1;
    double dn5_dt = -k_v * ((x[n + 1] - x[n - 1]) / (2 * delta_r)) + D * (x[n + 1] - 2 * x[n] + x[n - 1]) / pow(delta_r, 2);
    n += 1;
    double dn6_dt = -k_v * ((x[n + 1] - x[n - 1]) / (2 * delta_r)) + D * (x[n + 1] - 2 * x[n] + x[n - 1]) / pow(delta_r, 2);
    n += 1;
    double dn7_dt = -k_v * ((x[n + 1] - x[n - 1]) / (2 * delta_r)) + D * (x[n + 1] - 2 * x[n] + x[n - 1]) / pow(delta_r, 2);
    n += 1;
    double dn8_dt = -k_v * ((x[n + 1] - x[n - 1]) / (2 * delta_r)) + D * (x[n + 1] - 2 * x[n] + x[n - 1]) / pow(delta_r, 2);
    n += 1;
    double dn9_dt = -k_v * ((x[n + 1] - x[n - 1]) / (2 * delta_r)) + D * (x[n + 1] - 2 * x[n] + x[n - 1]) / pow(delta_r, 2);
    double n_k = dn9_dt;

    // Mean vacoule density for  department k all vacuoles above k in size are assumed constant size
    double r_k = r_0 + 8 * delta_r;
    double r_m = (r_0 + 10 * delta_r);

    // Calculating maximum vacuole volume department
    double dn_m_dt = k_v * n_k / (r_m - r_k) - mu_a * x[25];
    n_k = x[24];

    // mean vacuole
    double dphi_0_dt = ((mu_v * v_a1) / (K_v + s)) - k_v * x[15] * (3.141592653589793 * pow((r_0 + delta_0) , 3)) / 6;

    // Volume and Weight expressions
    double F_evp = x[4] * alpha_evp * (exp(2.5 * (x[7] - T0) / (Tv - T0)) - 1);
    double pho_feed = (c_s / 1000 * pho_g + (1 - c_s / 1000) * pho_w);

    // Dilution term
    double dilution = Fs + Fb + Fa + Fw - F_evp + Fpaa;

    // Change in Volume
    double dV1 = Fs + Fb + Fa + Fw + F_discharge / (pho_b / 1000) - F_evp + Fpaa;

    // Change in Weight
    double dWt = Fs * pho_feed / 1000 + pho_oil / 1000 * Foil + Fb + Fa + Fw + F_discharge - F_evp + Fpaa * pho_paa / 1000;

    // ODE's for Biomass regions
    double da_0_dt = r_b0 - r_d1 - x[11] * dilution / x[4];

    // Non growing regions
    double da_1_dt = r_e1 - r_b0 + r_d1 - (3.141592653589793 * pow((r_k + r_m), 3) / 6) * rho_d * k_v * n_k - x[12] * dilution / x[4];

    // Degenerated regions
    double da_3_dt = (3.141592653589793 * pow((r_k + r_m), 3) / 6) * rho_d * k_v * n_k - r_d4 - x[13] * dilution / x[4];

    // Autolysed regions
    double da_4_dt = r_d4 - x[14] * dilution / x[4];

    // Penicillin production
    double dP_dt = r_p - x[3] * dilution / x[4];

    // Active Biomass rate
    double X_1 = da_0_dt + da_1_dt + da_3_dt + da_4_dt;

    // Total biomass
    double X_t = x[11] + x[12] + x[13] + x[14];

    double Qrxn_X = X_1 * Y_QX * x[4] * Y_O2_X / 1000;
    double Qrxn_P = dP_dt * Y_QX * x[4] * Y_O2_P / 1000;

    double Qrxn_t = Qrxn_X + Qrxn_P;

    if (Qrxn_t < 0) {
        Qrxn_t = 0;
    }
    double N = RPM / 60;
    double D_imp = 2 * r_imp;
    double unaerated_power = (n_imp * Po * pho_b * pow(N, 3) * pow(D_imp, 5));
    double P_g = 0.706 * pow(((pow(unaerated_power, 2) * N * pow(D_imp, 3)) / pow(Fg, 0.56)), 0.45);
    double P_n = P_g / unaerated_power;
    double variable_power = (n_imp * Po * pho_b * pow(N, 3) * pow(D_imp, 5) * P_n) / 1000;


    // Substrate utilization
    dxdt[0] = -r_se1 - r_sb0 - r_m0 - r_m1 - (
            Y_sP * mu_p * rho_a0 * v_a1 * P_inhib * DO_2_inhib_P * PAA_inhib_P) + Fs * c_s / x[4] + Foil * c_oil /
                                                                                                    x[4] - x[0] * dilution / x[4];

    // Dissolved oxygen
    double V_s = Fg / (3.141592653589793 * pow(r, 2));
    double T = x[7];
    double V = x[4];
    double V_m = x[4] / 1000.0;
    double P_air = ((V_s * R * T * V_m / (22.4 * h_b)) * log(1 + pho_b * 9.81 * h_b / (pressure_top * 1e5)));
    double P_t1 = (variable_power + P_air);
    if (viscosity <= 4) {
        viscosity = 1;
    }
    double vis_scaled = viscosity / 100.0;
    double oil_f = Foil / V;
    double kla = alpha_kla * (pow(V_s, a) * pow((P_t1 / V_m), b) * pow(vis_scaled, c)) * (1 - pow(oil_f, d));
    double OUR = -X_1 * Y_O2_X - m_O2_X * X_t - dP_dt * Y_O2_P;
    double OTR = kla * (DOstar_tp - x[1]);
    dxdt[1] = OUR + OTR - (x[1] * dilution / x[4]);

    // O_2 off-gas
    double Vg = epsilon * V_m;
    double Qfg_in = 85714.28571428572 * Fg;
    double Qfg_out = Fg * (N2_in / (1 - x[2] - x[27] / 100)) * 85714.28571428572;
    dxdt[2] = (Qfg_in * O_2_in - Qfg_out * x[2] - 0.06 * OTR * V_m) / (Vg * 1293.3035714285716);

    // Penicillin production rate
    dxdt[3] = r_p - x[3] * dilution / x[4];

    // Volume change
    dxdt[4] = dV1;

    // Weight change
    dxdt[5] = dWt;

    // pH
    double pH_dis = Fs + Foil + Fb + Fa + F_discharge + Fw;

    double cb;
    double ca;
    double pH_balance;
    double x6 = x[6];

    if (-log10(x6) < 7) {
        cb = -abc;
        ca = abc;
        pH_balance = 0;
    } else {
        cb = abc;
        ca = -abc;
        x6 = (1e-14 / x6 - x6);
        pH_balance = 1;
    }

    // Calculation of ion addition
    double B = -(x6 * x[4] + ca * Fa * step1 + cb * Fb * step1) / (x[4] + Fb * step1 + Fa * step1);

    if (pH_balance == 1) {
        dxdt[6] = -gamma1 * (r_b0 + r_e1 + r_d4 + r_d1 + m_ph * total_X) - gamma1 * r_p - gamma2 * pH_dis + (
                (-B - pow((pow(B, 2) + 4e-14), .5)) / 2 - x6);
    }

    if (pH_balance == 0) {
        dxdt[6] = gamma1 * (r_b0 + r_e1 + r_d4 + r_d1 + m_ph * total_X) + gamma1 * r_p + gamma2 * pH_dis + (
                (-B + pow((pow(B, 2) + 4e-14),.5)) / 2 - x6);
    }

    // Temperature
    double Ws = P_t1;
    double Qcon = U_jacket * A_c * (x[7] - Tair);
    double dQ_dt = Fs * pho_feed * C_ps * (Tf - x[7]) / 1000 + Fw * pho_w * C_pw * (
            Tw - x[7]) / 1000 - F_evp * pho_b * C_pw / 1000 - dealta_H_evap * F_evp * pho_w / 1000 + Qrxn_t + Ws - (
                                                                                                                           alpha_1 / 1000) * pow(Fc, (beta_T + 1)) * (
                                                                                                                           (x[7] - Tcin) / (Fc / 1000 + (alpha_1 * pow((Fc / 1000), beta_T)) / 2 * pho_b * C_ps)) - (
                                                                                                                                                                                                                            alpha_1 / 1000) * pow(Fh, (beta_T + 1)) * (
                                                                                                                                                                                                                            (x[7] - Th) / (Fh / 1000 + (alpha_1 * pow((Fh / 1000), beta_T)) / 2 * pho_b * C_ps)) - Qcon;
    dxdt[7] = dQ_dt / ((x[4] / 1000) * C_pw * pho_b);

    // Heat generation
    dxdt[8] = dQ_dt;

    // Viscosity
    dxdt[9] = 3 * pow(a_0, (1 / 3)) * (1 / (1 + exp(-k1 * (t - t1)))) * (1 / (1 + exp(-k2 * (t - t2)))) - k3 * Fw;

    // Total X
    dxdt[10] = x[11] + x[12] + x[13] + x[14];

    //
    //   Adding in the ODE's for hyphae
    //
    dxdt[11] = da_0_dt;
    dxdt[12] = da_1_dt;
    dxdt[13] = da_3_dt;
    dxdt[14] = da_4_dt;
    dxdt[15] = dn0_dt;
    dxdt[16] = dn1_dt;
    dxdt[17] = dn2_dt;
    dxdt[18] = dn3_dt;
    dxdt[19] = dn4_dt;
    dxdt[20] = dn5_dt;
    dxdt[21] = dn6_dt;
    dxdt[22] = dn7_dt;
    dxdt[23] = dn8_dt;
    dxdt[24] = dn9_dt;
    dxdt[25] = dn_m_dt;
    dxdt[26] = dphi_0_dt;

    // CO_2
    double total_X_CO2 = x[11] + x[12];
    double CER = total_X_CO2 * q_co2 * V;
    dxdt[27] = (117857.14285714287 * Fg * C_CO2_in + CER - 117857.14285714287 * Fg * x[27]) / (Vg * 1293.3035714285716);

    // dissolved CO_2
    double Henrys_c_co2 = (exp(11.25 - 395.9 / (x[7] - 175.9))) / 4400;
    double C_star_CO2 = (total_pressure * x[27]) / Henrys_c_co2;
    dxdt[28] = kla * delta_c_0 * (C_star_CO2 - x[28]) - x[28] * dilution / x[4];

    // PAA
    dxdt[29] = Fpaa * PAA_c / V - Y_PAA_P * dP_dt - Y_PAA_X * X_1 - m_PAA * x[3] - x[29] * dilution / x[4];

    // N
    double X_C_nitrogen = (-r_b0 - r_e1 - r_d1 - r_d4) * Y_NX;
    double P_C_nitrogen = -dP_dt * Y_NP;
    dxdt[30] = (NH3_shots * N_conc_shot) / x[4] + X_C_nitrogen + P_C_nitrogen - m_N * total_X + (
            1 * N_conc_paa * Fpaa / x[4]) + N_conc_oil * Foil / x[4] - x[30] * dilution / x[4];
    dxdt[31] = mu_p;
    dxdt[32] = mu_e;
}

py::array_t<double> odeint_integrate(py::array_t<double> np_initial_state, py::array_t<double> np_params, double start_time, double end_time, double dt) {
    py::buffer_info init_state_info = np_initial_state.request();
    auto init_state_ptr = static_cast<double *>(init_state_info.ptr);

    py::buffer_info params_info = np_params.request();
    auto params_ptr = static_cast<double *>(params_info.ptr);

    state_type initial_state;

    for (int i = 0; i < init_state_info.shape[0]; i++) {
        initial_state[i] = init_state_ptr[i];
    }

    for (int i = 0; i < params_info.shape[0]; i++) {
        params[i] = params_ptr[i];
    }

    integrate( pensim_func, initial_state, start_time, end_time, dt);

    return py::array_t<double>(state_type::size(), initial_state.data());
}

PYBIND11_MODULE(fastodeint, m) {
    m.doc() = R"pbdoc(
        fastodeint - ODE solver C++ implementation just for PenSimPy
        -----------------------

        .. currentmodule:: fastodeint

        .. autosummary::
           :toctree: _generate

           integrate
    )pbdoc";

    m.def("integrate", &odeint_integrate);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
