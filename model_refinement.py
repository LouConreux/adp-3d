import argparse
import torch
import os
import numpy as np
import torch.nn.functional as F
import pickle
from chroma import Protein
from chroma import Chroma
from chroma.layers.structure.rmsd import CrossRMSD

from src.plots import plot_metric, save_trajectory, plot_rmsd_ca_vs_completeness, plot_SAXS_profile
from src.profile import compute_profile, fit_profile
from src.structure import FormFactorType
from src.utils import X_to_particles, read_exp_profile


def main(args):
    torch.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.outdir, exist_ok=True)

    print("Saving arguments")
    args_dict = vars(args)
    with open(f"{args.outdir}/config.txt", 'w') as file:
        for key in sorted(args_dict.keys()):
            file.write(f"{key}: {args_dict[key]}\n")

    print("Loading Chroma model")
    if args.weights_backbone is not None and args.weights_design is not None:
        chroma = Chroma(weights_backbone=args.weights_backbone, weights_design=args.weights_design)
    else:
        chroma = Chroma()
    backbone_network = chroma.backbone_network
    design_network = chroma.design_network
    def multiply_R(Z, C): return backbone_network.noise_perturb.base_gaussian._multiply_R(Z, C)
    def multiply_R_inverse(X, C): return backbone_network.noise_perturb.base_gaussian._multiply_R_inverse(X, C)
    def multiply_covariance(dU, C): return backbone_network.noise_perturb.base_gaussian.multiply_covariance(dU, C)

    print("Initializing Forward SAXS Profile Model")
    protein = Protein.from_PDB(args.pdb, device='cuda')
    X_gt, C_gt, S_gt = protein.to_XCS(all_atom=False)  # we use X_gt to compute RMSD
    mask_gt = (C_gt == 1)[0]
    n_residues = X_gt.shape[1]
    X_gt_full, C_gt_full, S_gt_full = protein.to_XCS(all_atom=True)
    chi_gt, mask_chi = design_network.X_to_chi(X_gt_full, C_gt_full, S_gt_full)

    print("Computing SAXS profile")
    exp_profile = read_exp_profile(args.dat)
    particles = X_to_particles(X_gt_full, C_gt_full, S_gt_full)
    model_profile = compute_profile(
        particles=X_to_particles(X_gt_full, C_gt_full, S_gt_full), min_q=exp_profile.min_q_, max_q=exp_profile.max_q_, delta_q=exp_profile.delta_q_, ff_type=FormFactorType.ALL_ATOMS,
    )

    print("Initializing backbone")
    C_gt = torch.abs(C_gt).expand(args.population_size, -1)
    S_gt = S_gt.expand(args.population_size, -1)
    chi_gt = chi_gt.expand(args.population_size, -1, -1)
    if args.init_gt:
        X = torch.clone(X_gt).expand(args.population_size, -1, -1, -1) + args.eps_init
        if args.std_dev_init > 1e-8:
            X += args.std_dev_init * torch.randn_like(X)
        Z = multiply_R_inverse(X, C_gt)
    else:
        Z = torch.randn(args.population_size, *X_gt.shape[1:]).float().cuda()
        X = multiply_R(Z, C_gt)
    V_s = torch.zeros_like(Z)
    V_c = torch.zeros_like(Z)
    V_p = torch.zeros_like(Z)
    
    def sample_chi(X, t=0):
        print("Sampling side chain angles")
        _X = F.pad(X, [0, 0, 0, 10])
        node_h, edge_h, edge_idx, mask_i, mask_ij = design_network.encode(_X, C_gt, t=t)
        permute_idx = design_network.traversal(_X, C_gt)
        _, chi_sample, _, logp_chi, _ = design_network.decoder_chi.decode(
            _X,
            C_gt,
            S_gt,
            node_h,
            edge_h,
            edge_idx,
            mask_i,
            mask_ij,
            permute_idx
        )
        return chi_sample
    if args.use_gt_chi:
        chi_sample = chi_gt
    else:
        chi_sample = torch.clone(sample_chi(X, t=1)).detach() if args.sample_chi_every > 0 else None

    def t_fn(epoch):
        if args.temporal_schedule == 'linear':
            return (-args.t + 0.001) * epoch / args.epochs + args.t
        elif args.temporal_schedule == 'sqrt':
            return (1.0 - 0.001) * (1. - np.sqrt(epoch / args.epochs)) + 0.001
        elif args.temporal_schedule == 'constant':
            return args.t
        else:
            raise NotImplementedError
    
    def profile_error(prof, prof_gt):
        profile, chi_square, fitted_params = fit_profile(prof_gt, prof, min_c1=0.9, max_c1=1.2, min_c2=-2.0, max_c2=4.0, use_offset=False)
        chi_square = torch.tensor(chi_square).float().cuda()
        return chi_square

    def get_gradient_Z_s(Z, t, epoch):
        if args.lr_sequence > 0. and epoch >= args.activate_sequence:
            with (torch.enable_grad()):
                Z.requires_grad_(True)
                _X = multiply_R(Z, C_gt)
                _X_input = F.pad(_X, [0, 0, 0, 10])
                out = design_network(_X_input, C_gt, S_gt, t.cuda())
                logp_S = out["logp_S"]
                loss_s = -logp_S.sum()
                loss_s.backward()
                grad_Z_s = Z.grad
            Z.requires_grad_(False)
        else:
            grad_Z_s = torch.zeros(*Z.shape).float().to(X.device)
            loss_s = torch.tensor([0.]).float().cuda()
        return grad_Z_s, loss_s

    def get_gradient_Z_c(Z):
        if args.lr_inter_ca > 0.:
            with (torch.enable_grad()):
                Z.requires_grad_(True)
                _X = multiply_R(Z, C_gt)
                distances = torch.linalg.norm(_X[:, 1:, 1] - _X[:, :-1, 1])
                loss_c = ((distances - 3.8) ** 2).sum()
                loss_c.backward()
                grad_Z_c = Z.grad
            Z.requires_grad_(False)
        else:
            grad_Z_c = torch.zeros(*Z.shape).float().to(X.device)
            loss_c = torch.tensor([0.]).float().cuda()
        return grad_Z_c, loss_c
    
    def get_gradient_Z_p(Z):
        if args.lr_profile > 0.:
            with (torch.enable_grad()):
                Z.requires_grad_(True)
                _X = multiply_R(Z, C_gt)
                if args.sample_chi_every > 0 and epoch + 1 % args.sample_chi_every == 0:
                    if args.use_gt_chi:
                        chi_sample = chi_gt
                    else:
                        chi_sample = sample_chi(_X, t)
                _X_full, _ = design_network.chi_to_X(_X, C_gt, S_gt, chi_sample)
                particles = X_to_particles(_X_full, C_gt, S_gt)
                profile = compute_profile(
                    particles=particles, min_q=exp_profile.min_q_, max_q=exp_profile.max_q_, delta_q=exp_profile.delta_q_, ff_type=FormFactorType.ALL_ATOMS,
                )
                loss_p = profile_error(profile, exp_profile)
                loss_p.backward()
                grad_Z_p = Z.grad
            Z.requires_grad_(False)
        else:
            grad_Z_p = torch.zeros(*Z.shape).float().to(X.device)
            loss_p = torch.tensor([0.]).float().cuda()
        return grad_Z_p, loss_p

    trajectory = [torch.clone(X_gt[:, mask_gt]).detach().cpu().numpy(),
                  (torch.clone(X).detach().cpu().numpy(), 'initial state')]
    
    metrics = {'epoch': [], 'rmsd': [], 't': [], 'loss_m': [], 'loss_d': [], 'rmsd_ca': [],
               'resolution': [], 'loss_s': [], 'loss_p': [], 'lr_density': [], 'loss_d_per_sample': [], 'sampling_rate': [],
               'loss_c': []}

    print("--- Optimization starts now ---")
    for epoch in range(args.epochs):
        t = torch.tensor(t_fn(epoch)).float().cuda()

        if args.use_diffusion:
            with torch.no_grad():
                X0 = backbone_network.denoise(X.detach(), C_gt, t)
        else:
            X0 = X
        Z0 = multiply_R_inverse(X0, C_gt)

        grad_Z_s, loss_s = get_gradient_Z_s(Z0, t, epoch)
        V_s = args.rho_sequence * V_s + args.lr_sequence * grad_Z_s

        grad_Z_c, loss_c = get_gradient_Z_c(Z0)
        V_c = args.rho_inter_ca * V_c + args.lr_inter_ca * grad_Z_c

        grad_Z_p, loss_p = get_gradient_Z_p(Z0)
        V_p = args.rho_profile * V_p + args.lr_profile * grad_Z_p

        Z0 = Z0 - V_s - V_c - V_p

        if args.use_diffusion:
            tm1 = torch.tensor(t_fn(epoch + 1)).float().cuda()
            alpha, sigma, _, _, _, _ = backbone_network.noise_perturb._schedule_coefficients(tm1)
            X = multiply_R(alpha * Z0 + sigma * torch.randn_like(Z0), C_gt)
        else:
            X = multiply_R(Z0, C_gt)

        if (epoch + 1) % args.log_every == 0:
            rmsds = []
            rmsds_cas = []
            for i in range(args.population_size):
                rmsd, _ = CrossRMSD().pairedRMSD(
                    torch.clone(X[i, mask_gt]).cpu().reshape(1, -1, 3),
                    torch.clone(X_gt[0, mask_gt]).cpu().reshape(1, -1, 3),
                    compute_alignment=True
                )
                rmsd_ca, _ = CrossRMSD().pairedRMSD(
                    torch.clone(X[i, mask_gt, 1, :]).cpu().reshape(1, -1, 3),
                    torch.clone(X_gt[0, mask_gt, 1, :]).cpu().reshape(1, -1, 3),
                    compute_alignment=True
                )
                rmsds.append(rmsd.item())
                rmsds_cas.append(rmsd_ca.item())
            idx_best = np.argmin(rmsds)
            rmsd_best = rmsds[idx_best]
            rmsd_ca_best = rmsds_cas[idx_best]

            rmsd, X_aligned = CrossRMSD().pairedRMSD(
                torch.clone(X[idx_best, mask_gt]).cpu().reshape(1, -1, 3),
                torch.clone(X_gt[0, mask_gt]).cpu().reshape(1, -1, 3),
                compute_alignment=True
            )
            X_aligned = X_aligned.reshape(1, -1, 4, 3)

            metrics['epoch'].append(epoch)
            metrics['t'].append(t.item())
            metrics['loss_s'].append(loss_s.item())
            metrics['loss_c'].append(loss_c.item())
            metrics['loss_p'].append(loss_p.item())
            metrics['rmsd'].append(rmsds)
            metrics['rmsd_ca'].append(rmsds_cas)
            trajectory.append((torch.clone(X[idx_best][None]).detach().cpu().numpy(), 'x-update'))
            print(f"Epoch {epoch + 1}/{args.epochs}, Loss Profile: {loss_p.item():.4e}, RMSD: {rmsd_best:.2e}, RMSD CA: {rmsd_ca_best:.2e}")

    C_gt = C_gt[0:1]
    S_gt = S_gt[0:1]
    X_full, _ = design_network.chi_to_X(X[idx_best][None], C_gt, S_gt, chi_sample)

    print(f"Saving {args.outdir}/metrics.pkl")
    with open(f"{args.outdir}/metrics.pkl", 'wb') as file:
        pickle.dump(metrics, file)

    print(f"Saving {args.outdir}/{args.outdir.split('/')[-1]}.pdb")
    protein_out = Protein.from_XCS(X_full, C_gt, S_gt)
    protein_out.to_PDB(f"{args.outdir}/{args.outdir.split('/')[-1]}.pdb")

    for key in metrics.keys():
        if key != 'epoch' and key != 'loss_d_per_sample':
            print(f"Saving {args.outdir}/{key}.png")
            plot_metric(metrics, key, f"{args.outdir}/{key}.png")

    print(f"Saving {args.outdir}/{args.outdir.split('/')[-1]}.mp4")
    save_trajectory(trajectory, f"{args.outdir}/{args.outdir.split('/')[-1]}.mp4")

    print(f"Saving {args.outdir}/rmsd_ca_vs_completeness.png")
    plot_rmsd_ca_vs_completeness(X_gt, X[idx_best][None], mask_gt.cpu(), f"{args.outdir}/rmsd_ca_vs_completeness.png")

    print(f"Saving {args.outdir}/SAXS_profile.png")
    particles = X_to_particles(X_full, C_gt, S_gt)
    model_profile = compute_profile(
        particles=particles, min_q=exp_profile.min_q_, max_q=exp_profile.max_q_, delta_q=exp_profile.delta_q_, ff_type=FormFactorType.ALL_ATOMS,
    )
    profile, chi_square, fitted_params = fit_profile(exp_profile, model_profile, min_c1=0.9, max_c1=1.2, min_c2=-2.0, max_c2=4.0, use_offset=False)
    plot_SAXS_profile(profile, exp_profile, chi_square, fitted_params, f"{args.outdir}/SAXS_profile.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # required parameters
    parser.add_argument('--outdir', type=str, required=True, help="Path to output directory.")
    parser.add_argument('--dat', type=str, required=True, help="Path to the SAXS data file.")
    parser.add_argument('--pdb', type=str, required=True, help="Path to deposited PDB file.")

    # I/O parameters
    parser.add_argument('--remove-oxt', type=int, default=1, help="Flag to ignore terminal oxygen.")
    parser.add_argument('--weights-backbone', type=str, default=None, help="Path to Chroma weights (backbone).")
    parser.add_argument('--weights-design', type=str, default=None, help="Path to Chroma weights (design).")
    parser.add_argument('--resolution', type=float, default=2.0, help="Resolution of the density map")
    parser.add_argument('--unpad-len', type=int, default=0, help="Number of (empty) voxels to remove on each side of the input voxel grid, to speed up computation.")

    # optimization parameters
    parser.add_argument('--epochs', type=int, default=4000, help="Number of epochs.")
    parser.add_argument('--population-size', type=int, default=1, help="Number of atomic models to simultaneously optimize.")
    parser.add_argument('--lr-sequence', type=float, default=1e-5, help="Learning rate for the sequence loss.")
    parser.add_argument('--rho-sequence', type=float, default=0.9, help="Momentum for the sequence loss.")
    parser.add_argument('--lr-inter-ca', type=float, default=0.0, help="Learning rate for the inter-CA loss.")
    parser.add_argument('--rho-inter-ca', type=float, default=0.9, help="Momentum for the inter-CA loss.")
    parser.add_argument('--lr-profile', type=float, default=1e-2, help="Learning rate for the intensity profile loss.")
    parser.add_argument('--rho-profile', type=float, default=0.9, help="Momentum for the intensity profile loss.")
    parser.add_argument('--activate-sequence', type=int, default=3000, help="Number of epochs before activating sequence loss.")

    # diffusion parameters
    parser.add_argument('--use-diffusion', type=int, default=1, help="Flag to use the diffusion model.")
    parser.add_argument('--temporal-schedule', type=str, default='sqrt', choices=['sqrt', 'linear', 'constant'], help="Type of temporal schedule.")
    parser.add_argument('--t', type=float, default=1.0, help="Initial diffusion time (between 0 and 1).")

    # random sampling
    parser.add_argument('--sampling-rate-schedule', type=str, default='constant', choices=['constant', 'linear', 'exp'], help='Type of schedule for the sampling rate.')
    parser.add_argument('--sampling-rate-start', type=float, default=0.1, help='Initial sampling rate.')
    parser.add_argument('--sampling-rate-end', type=float, default=1.0, help='Final sampling rate.')

    # side-chain parameters
    parser.add_argument('--sample-chi-every', type=int, default=100, help="Frequency (in epochs) of side-chain sampling.")
    parser.add_argument('--use-gt-chi', type=int, default=0, help="Flag to use ground truth side-chain angles, for debugging purposes.")
    
    # genetic parameters
    parser.add_argument('--replication-factor', type=int, default=2, help='Number of replications at each selection step.')
    parser.add_argument('--activate-replication', type=int, default=1, help="Number of epochs to wait before activating the selection/replication.")
    parser.add_argument('--select-best-every', type=int, default=500, help='Frequency (in epochs) of selection/replication (-1 to de-activate).')

    # initialization parameters
    parser.add_argument('--seed', type=int, default=0, help="Random seed.")
    parser.add_argument('--init-gt', type=int, default=0, help="Flag to initialize the model from the deposited CIF, for debugging purposes.")
    parser.add_argument('--std-dev-init', type=float, default=0.0, help="Intensity of Gaussian random noise added on ground truth.")
    parser.add_argument('--eps-init', type=float, default=0.0, help="Size of initial deviation to ground truth in the direction (1, 1, 1).")

    # logging parameters
    parser.add_argument('--log-every', type=int, default=10, help="Frequency (in epochs) for logging.")

    args = parser.parse_args()
    main(args)
