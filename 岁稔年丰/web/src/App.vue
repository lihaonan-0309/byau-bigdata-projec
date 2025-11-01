<script setup lang="ts">
</script>

<template>
  <RouterView />
</template>

<style lang="scss">
///
html,body{
    width: 100%;
    height: 100%;
}
#app{
    height: 100%;
    min-width: 1420px;
}
@media screen and (max-width: 1440px) {
  #app{
    min-width: 1260px;
  }
}

</style>
