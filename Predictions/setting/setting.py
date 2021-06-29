class Setting:

    def __init__(self, prefix, train_split, split_cases, add_end, percentage=None, train_k=None, filter_cases=None):
        self.prefixsize = prefix
        self.train_percentage = percentage
        self.train_k = train_k
        self.train_split = train_split
        self.split_cases = split_cases
        self.add_end = add_end
        self.filter_cases = filter_cases

    def __str__(self):
        if self.prefixsize:
            return "Prefixsize: %i, Train percentage: %i, Split_data: %r, Split_cases: %r, Add_end: %r" \
                   % (self.prefixsize, self.train_percentage, self.train_split, self.split_cases, self.add_end)
        else:
            return "Prefixsize: None, Train percentage: %i, Split_data: %r, Split_cases: %r, Add_end: %r" \
                   % (self.train_percentage, self.train_split, self.split_cases, self.add_end)

    def to_file_str(self):
        if self.prefixsize:
            return "Prefixsize: %i\nTrain percentage: %i\nSplit_data: %r\nSplit_cases: %r\nAdd_end: %r\n" \
                   % (self.prefixsize, self.train_percentage, self.train_split, self.split_cases, self.add_end)
        else:
            return "Prefixsize: Max\nTrain percentage: %i\nSplit_data: %r\nSplit_cases: %r\nAdd_end: %r\n" \
                   % (self.train_percentage, self.train_split, self.split_cases, self.add_end)