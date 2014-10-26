# A filler is used to fill (initialize) the layer parameters

abstract Filler

type ConstantFiller <: Filler
  value
end

function fill(filler::ConstantFiller, blob::Blob)
  fill!(blob.data, filler.value)
end
