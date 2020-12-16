namespace misc{
struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

int round(float x){
    return (int)(x+0.5);
}
}