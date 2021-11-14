from stric.datasets.yahoo_dataset import YahooDataset


if __name__ == "__main__":
    past_window_lenght, fut_window_lenght = 100, 1
    dataset = YahooDataset(
            past_len=past_window_lenght,
            fut_len=fut_window_lenght,
            seed= 0,
            data_path = 'data',
            dataset_subset = 'A1Benchmark',
            dataset_index = 'all',
    )

    dataset = M4Dataset(
            past_len=past_window_lenght,
            fut_len=fut_window_lenght,
            seed= 0,
            data_path = 'data',
            dataset_subset = 'Hourly',
            dataset_index = 1,
    )

    dataset = NABDataset(
            past_len=past_window_lenght,
            fut_len=fut_window_lenght,
            seed= 0,
            data_path = 'data',
            dataset_subset = 'realTweets',
            dataset_index = 'all',
    )

    dataset = SMDDataset(
            past_len=past_window_lenght,
            fut_len=fut_window_lenght,
            seed=0,
            data_path = 'data',
            dataset_subset = 'train',
            dataset_index = 0, #from 0 to 37,
            normalize=True,
        )
