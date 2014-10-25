abstract Backend

type CPU <: Backend; end
type GPU <: Backend; end


type Blob
    name :: String
    data
end
