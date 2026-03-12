
import mmcv
import numpy as np
from mmdet.datasets.builder import PIPELINES
import torch


@PIPELINES.register_module()
class LoadSuperPointsFromFile(object):
    """Load Points From File.

    Load superpoints points from file.

    Args:
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 multi_scale=1,
                 file_client_args=dict(backend='disk')):
      
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.multi_scale = multi_scale

    def _load_superpoints(self, spts_filename):
        """Private function to load superpoints data.

        Args:
            pts_filename (str): Filename of superpoints data.

        Returns:
            np.ndarray: An array containing superpoints data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            spts_bytes = self.file_client.get(spts_filename)
            superpoints = np.frombuffer(spts_bytes, dtype=np.int)
        except ConnectionError:
            mmcv.check_file_exist(spts_filename)
            superpoints = np.fromfile(spts_filename, dtype=np.long)
        return superpoints

    def __call__(self, results):
        """Call function to load superpoints data from file.

        Args:
            results (dict): Result dict containing superpointss data.

        Returns:
            dict: The result dict containing the superpoints data. \
                Added key and value are described below.
        """
        superpoints_filename = results['superpoints_filename']
        
        superpoints = self._load_superpoints(superpoints_filename)

        results['superpoints'] = superpoints
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ 
        return repr_str
    

@PIPELINES.register_module()
class LoadFusedPointsFromFile(object):
    """Load Fused Points From File.

    Load fused points from file.

    Args:
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 multi_scale=1,
                 file_client_args=dict(backend='disk')):
      
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.multi_scale = multi_scale

    def _load_fusedpoints(self, fpts_filename):
        """Private function to load fusedpoints data.

        Args:
            pts_filename (str): Filename of fusedpoints data.

        Returns:
            np.ndarray: An array containing fusedpoints data.
        """
        # if self.file_client is None:
        #     self.file_client = mmcv.FileClient(**self.file_client_args)
        # try:
        #     spts_bytes = self.file_client.get(fpts_filename)
        #     superpoints = np.frombuffer(spts_bytes, dtype=np.int)
        # except ConnectionError:
        #     mmcv.check_file_exist(fpts_filename)
        #     superpoints = np.fromfile(fpts_filename, dtype=np.long)
        
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        try:
            fpts_bytes = self.file_client.get(fpts_filename)
        # 先保存到临时 buffer 再用 torch.load 加载
            import io
            buffer = io.BytesIO(fpts_bytes)
            fused_data = torch.load(buffer, map_location='cpu')
        except ConnectionError:
            mmcv.check_file_exist(fpts_filename)
            fused_data = torch.load(fpts_filename, map_location='cpu')

        # 只取出 'feat' 部分
        fused_feats = fused_data['feat'][:, 3:]

        return fused_feats
        

    def __call__(self, results):
        """Call function to load fusedpoints data from file.

        Args:
            results (dict): Result dict containing fusedpoints data.

        Returns:
            dict: The result dict containing the fusedpoints data. \
        Args:
            results (dict): Result dict containing fusedpoints data.

        Returns:
            dict: The result dict containing the fusedpoints data. \
                Added key and value are described below.
        """
        fusedpoints_filename = results['fused_points_filename']

        fusedpoints = self._load_fusedpoints(fusedpoints_filename)

        results['fused_points'] = fusedpoints
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ 
        return repr_str

