"""
Notation:
----------
x is generally used for either the state or the mean of a gaussian. It should be clear from context which it is.
P is used about the state covariance
z is a single measurement
Z (capital) are mulitple measurements so that z = Z[k] at a given time step
v is the innovation z - h(x)
S is the innovation covariance
"""
# %% Imports
# types
from typing import Union, Any, Dict, Optional, List, Sequence, Tuple, Iterable
from typing_extensions import Final

# packages
from dataclasses import dataclass, field
import numpy as np
import scipy.linalg as la
import scipy

# local
import dynamicmodels as dynmods
import measurementmodels as measmods # measurmentmodels as measmods
from gaussparams import GaussParams, GaussParamList

# %% The EKF


@dataclass
class EKF:
    # A Protocol so duck typing can be used
    dynamic_model: dynmods.DynamicModel
    # A Protocol so duck typing can be used
    sensor_model: measmods.MeasurementModel

    #_MLOG2PIby2: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._MLOG2PIby2: Final[float] = self.sensor_model.m * \
            np.log(2 * np.pi) / 2

    def predict(self,
                ekfstate: GaussParams,
                # The sampling time in units specified by dynamic_model
                Ts: float,
                ) -> GaussParams:
        """Predict the EKF state Ts seconds ahead."""

        x, P = ekfstate  # tuple unpacking

        F = self.dynamic_model.F(x, Ts)
        Q = self.dynamic_model.Q(x, Ts)

        x_pred = self.dynamic_model.f(x,Ts) #F #None  # TODO
        P_pred = F @ P @ np.transpose(F) + Q #np.matmul(F,np.matmul(P,np.transpose(F))) + Q #None  # TODO

        state_pred = GaussParams(x_pred, P_pred)

        return state_pred

    def innovation_mean(
            self,
            z: np.ndarray,
            ekfstate: GaussParams,
            *,
            sensor_state: Dict[str, Any] = None,
    ) -> np.ndarray:
        """Calculate the innovation mean for ekfstate at z in sensor_state."""

        x = ekfstate.mean

        zbar = self.sensor_model.h(x, sensor_state=sensor_state)  #None  # TODO predicted measurement

        v = z-zbar #None  # TODO the innovation

        return v

    def innovation_cov(self,
                       z: np.ndarray,
                       ekfstate: GaussParams,
                       *,
                       sensor_state: Dict[str, Any] = None,
                       ) -> np.ndarray:
        """Calculate the innovation covariance for ekfstate at z in sensorstate."""

        x, P = ekfstate

        H = self.sensor_model.H(x, sensor_state=sensor_state)
        R = self.sensor_model.R(x, sensor_state=sensor_state, z=z)

        S = H @ P @ np.transpose(H) + R #np.matmul(H, np.matmul(P, np.transpose(H))) + R #None  # TODO the innovation covariance

        return S

    def innovation(self,
                   z: np.ndarray,
                   ekfstate: GaussParams,
                   *,
                   sensor_state: Dict[str, Any] = None,
                   ) -> GaussParams:
        """Calculate the innovation for ekfstate at z in sensor_state."""

        # TODO: reuse the above functions for the innovation and its covariance
        v = self.innovation_mean(z,ekfstate, sensor_state=sensor_state) #None
        S = self.innovation_cov(z,ekfstate, sensor_state=sensor_state) #None

        innovationstate = GaussParams(v, S)

        return innovationstate

    def update(self,
               z: np.ndarray,
               ekfstate: GaussParams,
               sensor_state: Dict[str, Any] = None
               ) -> GaussParams:
        """Update ekfstate with z in sensor_state"""

        x, P = ekfstate

        v, S = self.innovation(z, ekfstate, sensor_state=sensor_state)

        H = self.sensor_model.H(x, sensor_state=sensor_state)

        W =  P @ np.transpose(H) @ np.linalg.inv(S) #np.matmul(P ,np.matmul(np.transpose(H), np.linalg.inv(S))) # None  # TODO: the kalman gain, Hint: la.solve, la.inv

        x_upd = x + W @ v #x+np.matmul(W,v) #None  # TODO: the mean update

        P_upd = (np.eye(len(x))- W@H) @ P  #np.matmul((np.eye(np.size(x))-np.matmul(W,H))),P) #None  # TODO: the covariance update

        ekfstate_upd =  GaussParams(x_upd, P_upd)

        return ekfstate_upd

    def step(self,
             z: np.ndarray,
             ekfstate: GaussParams,
             # sampling time
             Ts: float,
             *,
             sensor_state: Dict[str, Any] = None,
             ) -> GaussParams:
        """Predict ekfstate Ts units ahead and then update this prediction with z in sensor_state."""

        # TODO: resue the above functions
        ekfstate_pred = self.predict(ekfstate, Ts) #None  # TODO
        ekfstate_upd = self.update(z, ekfstate_pred) #self.update(z, ekfstate) #None  # TODO
        return ekfstate_upd

    def NIS(self,
            z: np.ndarray,
            ekfstate: GaussParams,
            *,
            sensor_state: Dict[str, Any] = None,
            ) -> float:
        """Calculate the normalized innovation squared for ekfstate at z in sensor_state"""

        v, S = self.innovation(z, ekfstate, sensor_state=sensor_state)

        NIS = np.transpose(v) @ np.linalg.inv(S) @ v#np.matmul(np.transpose(v), np.matmul(np.inv(S),v)) #None  # TODO

        return NIS

    # @classmethod
    # def NEES(cls,
    #          ekfstate: GaussParams,
    #          # The true state to comapare against
    #          x_true: np.ndarray,
    #          ) -> float:
    #     """Calculate the normalized etimation error squared from ekfstate to x_true."""

    #     x, P = ekfstate

    #     x_diff = x - x_true #None  # Optional step
    #     NEES = np.transpose(x_diff)@np.linalg.inv(P)@x_diff  #np.matmul(np.transpose(x_diff), np.matmul(np.inv(P), x_diff)) #None  # TODO
    #     #P er singulær!
    #     return NEES

    def gate(self,
             z: np.ndarray,
             ekfstate: GaussParams,
             *,
             sensor_state: Dict[str, Any],
             gate_size_square: float,
             ) -> bool:
        """ Check if z is inside sqrt(gate_sized_squared)-sigma ellipse of ekfstate in sensor_state """

        # a function to be used in PDA and IMM-PDA
        gated = None  # TODO in PDA exercise
        return gated

    def loglikelihood(self,
                      z: np.ndarray,
                      ekfstate: GaussParams,
                      sensor_state: Dict[str, Any] = None
                      ) -> float:
        """Calculate the log likelihood of ekfstate at z in sensor_state"""
        # we need this function in IMM, PDA and IMM-PDA exercises
        # not necessary for tuning in EKF exercise
        v, S = self.innovation(z, ekfstate, sensor_state=sensor_state)
        nis_ = self.NIS(z, ekfstate)

        # TODO: log likelihood, Hint: log(N(v, S))) -> NIS, la.slogdet.
        ll = -0.5*(nis_ + la.slogdet(2*np.pi*S)) #TODO: Check here #None log of distribution 

        return ll

    @classmethod
    def estimate(cls, ekfstate: GaussParams):
        """Get the estimate from the state with its covariance. (Compatibility method)"""
        # dummy function for compatibility with IMM class
        return ekfstate

    def estimate_sequence(
            self,
            # A sequence of measurements
            Z: Sequence[np.ndarray],
            # the initial KF state to use for either prediction or update (see start_with_prediction)
            init_ekfstate: GaussParams,
            # Time difference between Z's. If start_with_prediction: also diff before the first Z
            Ts: Union[float, Sequence[float]],
            *,
            # An optional sequence of the sensor states for when Z was recorded
            sensor_state: Optional[Iterable[Optional[Dict[str, Any]]]] = None,
            # sets if Ts should be used for predicting before the first measurement in Z
            start_with_prediction: bool = False,
    ) -> Tuple[GaussParamList, GaussParamList]:
        """Create estimates for the whole time series of measurements."""

        # sequence length
        K = len(Z)

        # Create and amend the sampling array
        Ts_start_idx = int(not start_with_prediction)
        Ts_arr = np.empty(K)
        Ts_arr[Ts_start_idx:] = Ts
        # Insert a zero time prediction for no prediction equivalence
        if not start_with_prediction:
            Ts_arr[0] = 0

        # Make sure the sensor_state_list actually is a sequence
        sensor_state_seq = sensor_state or [None] * K

        # initialize and allocate
        ekfupd = init_ekfstate
        n = init_ekfstate.mean.shape[0]
        ekfpred_list = GaussParamList.allocate(K, n)
        ekfupd_list = GaussParamList.allocate(K, n)
        # perform the actual predict and update cycle
        # TODO loop over the data and get both the predicted and updated states in the lists
        # the predicted is good to have for evaluation purposes
        # A potential pythonic way of looping through  the data
        
        #x0, P0 = self.predict(init_ekfstate, Ts)
        
        for k, (zk, Tsk, ssk) in enumerate(zip(Z, Ts_arr, sensor_state_seq)):
        #     ekfpred = self.predict(ekfupd,Tsk)
        #     ekfpred_list[k] = ekfpred

        #     ekfupd = self.update(zk,ekfpred,ssk)
        #     ekfupd_list[k] = ekfupd
        # return ekfpred_list, ekfupd_list
            if k == 0:
                #x_pred, P_pred = self.predict(ekfupd,Tsk)
                ekfpred = self.predict(ekfupd,Tsk)
            else:
                #x_pred, P_pred = self.predict(ekfpred_list[k-1],Tsk)
                ekfpred = self.predict(ekfupd,Tsk)
            #x_upd , P_upd = self.update(zk, GaussParams(x_pred,P_pred),ssk)
            ekfupd = self.update(zk, ekfpred, ssk)
            ekfpred_list[k] = ekfpred#GaussParams(x_pred, P_pred)
            ekfupd_list[k] = ekfupd#GaussParams(x_upd, P_upd)
        return ekfpred_list, ekfupd_list

    def performance_stats(
            self,
            *,
            z: Optional[np.ndarray] = None,
            ekfstate_pred: Optional[GaussParams] = None,
            ekfstate_upd: Optional[GaussParams] = None,
            sensor_state: Optional[Dict[str, Any]] = None,
            x_true: Optional[np.ndarray] = None,
            # None: no norm, -1: all idx, seq: a single norm for given idxs, seqseq: a norms for idxseq
            norm_idxs: Optional[Iterable[Sequence[int]]] = None,
            # The sequence of norms to calculate for idxs, see np.linalg.norm ord argument.
            norms: Union[Iterable[int], int] = 2,
    ) -> Dict[str, Union[float, List[float]]]:
        """Calculate performance statistics available from the given parameters."""
        stats: Dict[str, Union[float, List[float]]] = {}

        # NIS, needs measurements
        if z is not None and ekfstate_pred is not None:
            stats['NIS'] = self.NIS(
                z, ekfstate_pred, sensor_state=sensor_state)

        # NEES and RMSE, needs ground truth
        if x_true is not None:
            # prediction
            if ekfstate_pred is not None:
                #stats['NEESpred'] = self.NEES(ekfstate_pred, x_true)

                # distances
                err_pred = ekfstate_pred.mean - x_true
                if norm_idxs is None:
                    stats['dist_pred'] = np.linalg.norm(err_pred, ord=norms)
                elif isinstance(norm_idxs, Iterable) and isinstance(norms, Iterable):
                    stats['dists_pred'] = [
                        np.linalg.norm(err_pred[idx], ord=ord)
                        for idx, ord in zip(norm_idxs, norms)]

            # update
            if ekfstate_upd is not None:
                #stats['NEESupd'] = self.NEES(ekfstate_upd, x_true)

                # distances
                err_upd = ekfstate_upd.mean - x_true
                if norm_idxs is None:
                    stats['dist_upd'] = np.linalg.norm(err_upd, ord=norms)
                elif isinstance(norm_idxs, Iterable) and isinstance(norms, Iterable):
                    stats['dists_upd'] = [
                        np.linalg.norm(err_upd[idx], ord=ord)
                        for idx, ord in zip(norm_idxs, norms)]
        return stats

    def performance_stats_sequence(
            self,
            # Sequence length
            K: int,
            *,
            # The measurements
            Z: Optional[Iterable[np.ndarray]] = None,
            ekfpred_list: Optional[Iterable[GaussParams]] = None,
            ekfupd_list: Optional[Iterable[GaussParams]] = None,
            # An optional sequence of all the sensor states when Z was recorded
            sensor_state: Optional[Iterable[Optional[Dict[str, Any]]]] = None,
            # Optional ground truth for error checking
            X_true: Optional[Iterable[Optional[np.ndarray]]] = None,
            # Indexes to be used to calculate errornorms, multiple to separate the state space.
            # None: all idx, Iterable (eg. list): each element is an index sequence into the dimension of the state space.
            norm_idxs: Optional[Iterable[Sequence[int]]] = None,
            # The sequence of norms to calculate for idxs (see numpy.linalg.norm ord argument).
            norms: Union[Iterable[int], int] = 2,
    ) -> np.ndarray:
        """Get performance metrics on a pre-estimated sequence"""

        None_list = [None] * K

        for_iter = []
        for_iter.append(Z if Z is not None else None_list)
        for_iter.append(ekfpred_list or None_list)
        for_iter.append(ekfupd_list or None_list)
        for_iter.append(sensor_state or None_list)
        for_iter.append(X_true if X_true is not None else None_list)

        stats = []
        for zk, ekfpredk, ekfupdk, ssk, xtk in zip(*for_iter):
            stats.append(
                self.performance_stats(
                    z=zk, ekfstate_pred=ekfpredk, ekfstate_upd=ekfupdk, sensor_state=ssk, x_true=xtk,
                    norm_idxs=norm_idxs, norms=norms
                )
            )

        # make structured array
        dtype = [(key, *((type(val[0]), len(val)) if isinstance(val, Iterable)
                         else (type(val),))) for key, val in stats[0].items()]
        stats_arr = np.array([tuple(d.values()) for d in stats], dtype=dtype)

        return stats_arr


# %% End
