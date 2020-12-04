class Setting:

    def __init__(self, prefix, percentage, split_data, split_cases, add_end):
        self.prefixsize = prefix
        self.train_percentage = percentage
        self.split_data = split_data
        self.split_cases = split_cases
        self.add_end = add_end

    def __str__(self):
        return "Prefixsize: %i, Train percentage: %i, Split_data: %r, Split_cases: %r, Add_end: %r" \
               % (self.prefixsize, self.train_percentage, self.split_data, self.split_cases, self.add_end)

    def to_file_str(self):
        if self.prefixsize:
            return "Prefixsize: %i\nTrain percentage: %i\nSplit_data: %r\nSplit_cases: %r\nAdd_end: %r\n" \
                   % (self.prefixsize, self.train_percentage, self.split_data, self.split_cases, self.add_end)
        else:
            return "Prefixsize: Max\nTrain percentage: %i\nSplit_data: %r\nSplit_cases: %r\nAdd_end: %r\n" \
                   % (self.train_percentage, self.split_data, self.split_cases, self.add_end)